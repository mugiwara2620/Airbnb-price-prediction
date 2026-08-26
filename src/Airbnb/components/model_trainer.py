from torch import nn
import torch

# CTEATE THE PREDICT MODEL


class TabularNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.price_prediction = nn.Sequential(
            nn.Linear(in_features=19, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, x):
        return self.price_prediction(x)


# 2. Train the model
class ModelTrainer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initiate_model_trainer(self, data):
        X_train, y_train = train_data[], train_data[]
        

# CONVERT THE DATA TO
