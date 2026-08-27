import os
import pandas as pd
from src.Airbnb.utils import split_data
from src.Airbnb.components.model_trainer import ModelTrainer
from src.Airbnb.components.Data_transformation import DataTransformation


def test_training_model():
    print("Starting Model Training...")

    # 1. Initialize components
    data_transformation = DataTransformation()
    model_trainer = ModelTrainer()

    # 2. Load and split raw data into DataFrames
    file_path = os.path.join("Artifacts", "raw_data.csv")
    train_df, test_df = split_data(file_path)

    # 3. Clean 'amenities' strings into numeric feature counts
    if "amenities" in train_df.columns:
        train_df["amenities"] = train_df["amenities"].apply(
            lambda x: len(str(x).split(","))
        )
        test_df["amenities"] = test_df["amenities"].apply(
            lambda x: len(str(x).split(","))
        )

    # 4. Transform Data (returns fully numeric NumPy arrays)
    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiale_data_transformer(
            train_df=train_df, test_df=test_df
        )
    )

    print(f"Data transformed successfully. Preprocessor saved at: {preprocessor_path}")

    # 5. Initiate PyTorch Model Training
    r2_score = model_trainer.initiate_model_trainer(
        train_array=train_arr, test_array=test_arr
    )
    print(f"--- Training Complete --- Final R2 Score: {r2_score}")


if __name__ == "__main__":
    test_training_model()
