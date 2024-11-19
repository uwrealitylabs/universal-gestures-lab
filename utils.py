# Load environment variables
load_dotenv()

def log_training_metrics(auc_pr, auc_roc, final_loss, model_type):
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