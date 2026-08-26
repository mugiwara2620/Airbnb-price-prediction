import os
import sys
import pickle
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

from src.Airbnb.exception import customexception


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customexception(e, sys)


def split_data(file_path):

    df = pd.read_csv(file_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Total : {len(df)} | Train : {len(train_df)} | Test : {len(test_df)}")

    return train_df.to_numpy(), test_df.to_numpy()
