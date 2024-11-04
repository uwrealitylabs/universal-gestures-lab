import preprocess
import process_data
import model

def main():
    # Normalizes data and saves to data/normalized folder
    preprocess.main()
    # Splits data and converts them to PyTorch tensors
    process_data.main()
    # Runs model
    model.main()
    
if __name__ == "__main__":
    main()
