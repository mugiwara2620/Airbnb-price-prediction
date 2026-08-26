import os, sys, torch
import pandas as pd
import numpy as np

from src.Airbnb.components.model_trainer import TabularNN
from src.Airbnb.components.Data_transformation import DataTransformation
from src.Airbnb.utils import save_object, load_object, split_data


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("Artifacts", "Preprocessor.pkl")
        self.model_path = os.path.join("Artifacts", "model.pkl")

    def predict(self, features):
        model = TabularNN()
        model.load_state_dict(torch.load(self.model_path))

        model.eval()
        with torch.inference_mode():
            pred_price = model(features).item()
        return pred_price
