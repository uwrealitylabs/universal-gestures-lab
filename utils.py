import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from pathlib import Path
import os

def log_training_metrics(auc_pr, auc_roc, final_loss, model_type):
    # Load environment variables
    load_dotenv()
    """Logs training metrics to the NeonDB database."""
    conn = None
    try:
        db_url = os.getenv("NEONDB_URL")
        conn = psycopg2.connect(db_url, sslmode='require')
        cursor = conn.cursor()

        # Ensure data types are Python-native
        auc_pr = float(auc_pr)  # Convert to native float
        auc_roc = float(auc_roc)  # Convert to native float
        final_loss = float(final_loss)  # Convert to native float

        '''
        # Define the maximum number of rows allowed in the table
        MAX_ROWS = 1000000  # Adjust this value based on the NeonDB capacity

        # Check the current number of rows in the table
        cursor.execute("SELECT COUNT(*) FROM public.model_tracking;")
        row_count = cursor.fetchone()[0]

        if row_count >= MAX_ROWS:
            # Number of oldest rows to delete to free up space
            N = 100  # Adjust this value as needed

            # Delete the oldest N rows based on the datetime column
            delete_query = """
            DELETE FROM public.model_tracking
            WHERE ctid IN (
                SELECT ctid FROM public.model_tracking
                ORDER BY datetime ASC
                LIMIT %s
            );
            """
            cursor.execute(delete_query, (N,))
            conn.commit()
            print(f"Deleted {N} oldest rows to free up space.")
        '''

        query = """
        INSERT INTO public.model_tracking (datetime, auc_pr, auc_roc, final_loss, model_type)
        VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(query, (datetime.now(), auc_pr, auc_roc, final_loss, model_type))
        conn.commit()
        print("Training metrics logged successfully.")

    except psycopg2.Error as e:
        # Check if the error is due to storage capacity
        if 'could not extend file' in str(e).lower() or 'no space left on device' in str(e).lower():
            print("Storage capacity reached. Deleting oldest entries to free up space.")
            try:
                # Number of oldest rows to delete
                N = 100  # Adjust as needed

                # Delete the oldest N rows based on the datetime column
                delete_query = """
                DELETE FROM public.model_tracking
                WHERE ctid IN (
                    SELECT ctid FROM public.model_tracking
                    ORDER BY datetime ASC
                    LIMIT %s
                );
                """
                cursor.execute(delete_query, (N,))
                conn.commit()
                print(f"Deleted {N} oldest rows to free up space.")

                # Retry the insertion
                cursor.execute(query, (datetime.now(), auc_pr, auc_roc, final_loss, model_type))
                conn.commit()
                print("Training metrics logged successfully after freeing up space.")
            except Exception as del_e:
                print("Error during deletion or reinsertion:", del_e)
        else:
            print("Error logging training metrics:", e)

    except Exception as e:
        print("Error logging training metrics:", e)
    finally:
        if conn:
            cursor.close()
            conn.close()

def upload_to_b2(local_path, b2_file_name):
    # Uploads files to Backblaze B2 bucket and manages storage limits
    # Load environment variables
    load_dotenv()

    try:
        # Initialize B2 API
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", os.getenv("B2_KEY_ID"), os.getenv("B2_APPLICATION_KEY"))
        bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

        # Check current bucket usage
        MAX_STORAGE_GB = 10  # Backblaze free tier limit
        WARNING_THRESHOLD_GB = 9.5  # Warn if usage exceeds this
        total_usage_bytes = 0

        for file_version, _ in bucket.ls(latest_only=True):
            total_usage_bytes += file_version.size

        total_usage_gb = total_usage_bytes / (1024 ** 3)  # Convert bytes to GB
        print(f"Current bucket usage: {total_usage_gb:.2f} GB")

        if total_usage_gb >= MAX_STORAGE_GB:
            raise Exception("Storage limit exceeded! Free up space to continue.")
        elif total_usage_gb >= WARNING_THRESHOLD_GB:
            print("WARNING: Bucket storage is nearing its limit. Freeing up space...")
            free_up_space(bucket, total_usage_gb - WARNING_THRESHOLD_GB)

        # Upload the file
        local_file = Path(local_path).resolve()
        bucket.upload_local_file(local_file=local_file, file_name=b2_file_name)
        print(f"File uploaded to Backblaze B2: {b2_file_name}")

    except Exception as e:
        print("Error uploading file to Backblaze B2:", e)


def free_up_space(bucket, space_needed_gb):
    """Deletes the oldest files to free up space in the bucket."""
    space_needed_bytes = space_needed_gb * (1024 ** 3)  # Convert GB to bytes
    freed_space_bytes = 0

    # List all files, sorted by upload time (oldest first)
    files = list(bucket.ls(latest_only=True))
    files_sorted = sorted(files, key=lambda f: f[0].upload_timestamp)

    for file_version, _ in files_sorted:
        if freed_space_bytes >= space_needed_bytes:
            break

        print(f"Deleting file: {file_version.file_name} ({file_version.size / (1024 ** 2):.2f} MB)")
        bucket.delete_file_version(file_version.id_, file_version.file_name)
        freed_space_bytes += file_version.size

    print(f"Freed up space: {freed_space_bytes / (1024 ** 3):.2f} GB")