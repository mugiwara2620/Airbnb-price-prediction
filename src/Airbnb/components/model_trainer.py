import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.Airbnb.logger import logging
from src.Airbnb.utils import save_object
from src.Airbnb.exception import customexception


# 1. Define Network with Dynamic Input Dimension
class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.price_prediction = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.price_prediction(x)


# 2. Model Trainer Component
class ModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join("Artifacts", "model.pt")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Slice NumPy features and target columns BEFORE converting to PyTorch objects
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Convert object/raw array to float32 NumPy array first
            X_train = np.asarray(X_train, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.float32)

            X_test = np.asarray(X_test, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.float32)

            # Now convert to PyTorch Tensors
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            # Wrap into TensorDataset & DataLoader
            train_dataset = TensorDataset(X_train_t, y_train_t)
            test_dataset = TensorDataset(X_test_t, y_test_t)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Instantiate model dynamically with actual transformed features shape
            input_dim = X_train.shape[1]
            model = TabularNN(input_dim=input_dim).to(self.device)

            loss_fn = nn.L1Loss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

            epochs = 100
            for epoch in tqdm(range(epochs)):
                # --- Training ---
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = loss_fn(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # --- Testing ---
                model.eval()
                test_loss = 0.0
                with torch.inference_mode():
                    for batch_x, batch_y in test_loader:
                        batch_x, batch_y = (
                            batch_x.to(self.device),
                            batch_y.to(self.device),
                        )
                        test_preds = model(batch_x)
                        test_loss += loss_fn(test_preds, batch_y).item()

                test_loss /= len(test_loader)

                if epoch % 10 == 0:
                    logging.info(
                        f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                    )
                    print(
                        f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                    )

            model.eval()
            y_true = []
            y_preds = []

            with torch.inference_mode():
                for X_batch, y_batch in test_loader:
                    batch_x = X_batch.to(self.device)
                    test_preds = model(batch_x)
                    y_preds.extend(test_preds.cpu().numpy())
                    y_true.extend(y_batch.numpy())

            rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_preds))
            mae = mean_absolute_error(y_true=y_true, y_pred=y_preds)
            r2 = r2_score(y_true=y_true, y_pred=y_preds)
            print(f"Final Test Metrics -> R2: {r2:.4f} | RMSE: {rmse} | MAE: {mae}")
            # Export weights dict
            torch.save(model.state_dict(), self.model_path)
            logging.info(f"Model state saved successfully to {self.model_path}")
            return test_loss

        except Exception as e:
            raise customexception(e, sys)
