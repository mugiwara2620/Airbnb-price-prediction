# test_training_model.py
from src.Airbnb.components.Data_transformation import DataTransformation
from src.Airbnb.components.model_trainer import ModelTrainer


def test_training_model():
    print("Starting Model Training...")

    # 1. First preprocess raw CSV features into scaled/one-hot numeric arrays
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(
        train_path="Artifacts/train.csv", test_path="Artifacts/test.csv"
    )

    # 2. Pass numerical arrays into trainer
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(
        train_array=train_array, test_array=test_array
    )
    print(f"Training Complete! R2 Score: {r2_score}")


if __name__ == "__main__":
    test_training_model()
