import sys
import os

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.Airbnb.exception import customexception
from src.Airbnb.logger import logging
from src.Airbnb.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("Artifacts", "Preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Create and returns the Scikit-learn ColumnTransformer pipeline.
        """
        try:
            # Categorical and Numerical feature groups
            categorical_cols = [
                "property_type",
                "room_type",
                "bed_type",
                "cancellation_policy",
                "city",
                "cleaning_fee",
                "host_has_profile_pic",
                "host_identity_verified",
                "instant_bookable",
            ]
            numerical_cols = [
                "bedrooms",
                "beds",
                "amenities",
                "accommodates",
                "bathrooms",
                "host_response_rate",
                "latitude",
                "longitude",
                "number_of_reviews",
                "review_scores_rating",
            ]
            # Numerical Pipeline: Impute missing values with median, then scale
            num_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical Pipeline: Impute missing with most frequent, then One-Hot Encode
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                ]
            )

            return preprocessor
        except Exception as e:
            raise customexception(e, sys)

    def initiale_data_transformer(self, train_df, test_df):

        try:
            logging.info("Loaded train and test data successfully")

            # -----------------------------------
            # 1. Handle amenities BEFORE splitting
            # -----------------------------------

            def count_amenities(x):
                if pd.isna(x):
                    return 0

                return len(str(x).split(","))

            train_df = train_df.copy()
            test_df = test_df.copy()

            train_df["amenities"] = train_df["amenities"].apply(count_amenities)
            test_df["amenities"] = test_df["amenities"].apply(count_amenities)

            # -----------------------------------
            # 2. Create preprocessor
            # -----------------------------------

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "log_price"

            # -----------------------------------
            # 3. Separate X and y
            # -----------------------------------
            def convert_percentage(x):
                if pd.isna(x):
                    return np.nan

                x = str(x).strip()

                if x.endswith("%"):
                    return float(x[:-1]) / 100

                return float(x)

            train_df["host_response_rate"] = train_df["host_response_rate"].apply(
                convert_percentage
            )

            test_df["host_response_rate"] = test_df["host_response_rate"].apply(
                convert_percentage
            )

            input_feature_train_df = train_df.drop(columns=[target_column_name])

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])

            target_feature_test_df = test_df[target_column_name]

            # -----------------------------------
            # 4. Transform
            # -----------------------------------

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # -----------------------------------
            # 5. Combine X and y
            # -----------------------------------

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # -----------------------------------
            # 6. Save preprocessor
            # -----------------------------------

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info(
                f"Saved Preprocessor object to "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise customexception(e, sys)
