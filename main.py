from src.data_preprocessing import generate_data, preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate

def main():
    print("Generating data...")
    data = generate_data()

    print("Preprocessing data...")
    data = preprocess_data(data)

    print("Training model...")
    model, X_test, y_test = train_model(data)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()