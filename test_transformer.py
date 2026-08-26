import os
from src.Airbnb.components.Data_transformation import DataTransformation


def test_pipeline():

    train_path = os.path.join("Artifacts", "train_data.csv")
    test_path = os.path.join("Artifacts", "test_data.csv")

    print("Starting Data Transformer test")
    data_transformer = DataTransformation()
    test_arr, train_arr, preprocessor_path = data_transformer.initiale_data_transformer(
        train_path=train_path, test_path=test_path
    )

    print("---- SUCCESS ---")
    print(f"Preprocessor object saved at: {preprocessor_path}")
    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape: {test_arr.shape}")
    print(f"Sample transformed target value: {train_arr[0, -1]}")


if __name__ == "__main__":
    test_pipeline()
