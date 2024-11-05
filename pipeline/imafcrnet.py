import torch
import torch.nn as nn

class TARnetIMA_CFRClassifier(nn.Module):
    def __init__(self, input_dim, reg_l2, hidden_dim, n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Modified output layers for classification
        self.y0_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.n_classes)
        )

        self.y1_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.n_classes)
        )
        
    def forward_treated(self, treated):
        phi = self.phi(treated)
        y1 = self.y1_hidden(phi)
        return torch.sigmoid(y1)  # Apply log_softmax
    
    def forward_control(self, control):
        phi = self.phi(control)
        y0 = self.y0_hidden(phi)
        return torch.sigmoid(y0)  # Apply log_softmax
    
    def forward(self, x):  
        y1 = self.forward_treated(x)
        y0 = self.forward_control(x)
                
        # return estimation under treatment (y1) and estimation under control (y0)
        return y1, y0

