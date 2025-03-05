"""
model.py - Neural Network for Low Energy Nuclear Recoil Investigation (LENRI)
"""

import torch.nn as nn

class LENRI_CF4_1(nn.Module):
    """
    LENRI hyperparameter tuned model for CF4-1 dataset.
    """
    def __init__(self):
        super(LENRI_CF4_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 96),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0864),
            nn.Linear(96, 62),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0864),
            nn.Linear(62, 2),
        )

    def forward(self, x):
        return self.model(x)
    

class LENRI_Ar_CF4_1(nn.Module):
    """
    LENRI hyperparameter tuned model for Ar-CF4-1 dataset.
    """
    def __init__(self):
        super(LENRI_Ar_CF4_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 117),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0693),
            nn.Linear(117, 40),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0693),
            nn.Linear(40, 48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0693),
            nn.Linear(48, 3),
        )

    def forward(self, x):
        return self.model(x)
    
    
class LENRI_CF4_2(nn.Module):
    """
    LENRI hyperparameter tuned model for CF4-2 dataset with optimized hyperparameters.
    """
    def __init__(self):
        super(LENRI_CF4_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(18, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.09),
            nn.Linear(64, 48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.09),
            nn.Linear(48, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.09),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)
    

class LENRI_Ar_CF4_2(nn.Module):
    """
    LENRI hyperparameter tuned model for Ar-CF4-2 dataset.
    """
    def __init__(self):
        super(LENRI_Ar_CF4_2, self).__init__()
        hidden_layers = [64, 48, 32]
        self.model = nn.Sequential(
            nn.Linear(18, hidden_layers[0]),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.08408783486673037),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.08408783486673037),
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.08408783486673037),
            nn.Linear(hidden_layers[2], 3),
        )

    def forward(self, x):
        return self.model(x)
