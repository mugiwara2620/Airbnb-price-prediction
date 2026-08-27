import os, sys, torch
import pandas as pd
import numpy as np

from src.Airbnb.components.model_trainer import TabularNN
from src.Airbnb.components.Data_transformation import DataTransformation
from src.Airbnb.utils import save_object, load_object, split_data
from src.Airbnb.exception import customexception


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("Artifacts", "Preprocessor.pkl")
        self.model_path = os.path.join("Artifacts", "model.pt")

    def predict(self, features):
        try:
            # Preprocess features
            preprocessor = load_object(self.preprocessor_path)
            scaled_features = preprocessor.transform(features)

            # Convert to Pytorch Tensor
            tensor_features = torch.tensor(scaled_features, dtype=torch.float32)

            # Load model
            input_dim = scaled_features.shape[1]
            model = TabularNN(input_dim)

            model.load_state_dict(torch.load(self.model_path))
            model.eval()

            with torch.inference_mode():
                log_price_pred = model(tensor_features).numpy().flatten()

            price_pred = np.expm1(log_price_pred)
            return price_pred

        except Exception as e:
            raise customexception(e, sys)
