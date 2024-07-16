import process_data
import model

def main():
    # Splits data and converts them to PyTorch tensors
    process_data.main()
    # Runs model
    model.main()
    
if __name__ == "__main__":
    main()
