import os
import pandas as pd
from src.Airbnb.utils import split_data
from src.Airbnb.components.Data_transformation import DataTransformation


def test_pipeline():
    data_path = os.path.join("Artifacts", "raw_data.csv")

    train_df, test_df = split_data(file_path=data_path)

    print("Starting Data Transformer test")
    data_transformer = DataTransformation()
    test_arr, train_arr, preprocessor_path = data_transformer.initiale_data_transformer(
        train_df=train_df, test_df=test_df
    )

    print("---- SUCCESS ---")
    print(f"Preprocessor object saved at: {preprocessor_path}")
    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape: {test_arr.shape}")
    print(f"Sample transformed target value: {train_arr[0, -1]}")
    print(f"Sample data samples: {pd.DataFrame(train_arr[:5])}")


if __name__ == "__main__":
    test_pipeline()
