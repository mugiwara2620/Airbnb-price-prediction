import os
import pandas as pd
from src.Airbnb.utils import split_data
from src.Airbnb.components.model_trainer import ModelTrainer


def test_training_model():

    print("Starting Model Training")
    model_trainer = ModelTrainer()
    file_path = os.path.join("Artifacts", "raw_data.csv")

    train_array, test_array = split_data(file_path)
    model_trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)


if __name__ == "__main__":
    test_training_model()
