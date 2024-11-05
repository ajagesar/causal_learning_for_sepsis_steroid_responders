from utils import ROOT, to_tensor, device
#from src.TARNetCodeBase import TARnetIMA_CFRClassifier
from imafcrnet import TARnetIMA_CFRClassifier

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from sklearn.metrics import roc_auc_score, brier_score_loss

import logging
logger = logging.getLogger('run')
class Mixin:
    def hyperparameter_tuning(self): # TODO ranking of number of features to take into account as hyperparameter

        print("")
        print("using following for training:")
        print(device)
        print("")

        def objective(trial):

            if self.metalearner_classifier == 'XGB':
                train = self.preprocessed_data_scaled.loc[self.preprocessed_data_scaled[self.label_id].isin(self.train_ids)].copy()
            if self.metalearner_classifier != 'XGB':
                train = self.preprocessed_data_scaled_imputed.loc[self.preprocessed_data_scaled_imputed[self.label_id].isin(self.train_ids)].copy()

            train_for_tuning = train.sample(frac=0.8, random_state=self.random_state)
            train_for_valid = train.drop(train_for_tuning.index)

            if self.algorithm == "tarnet-imacfr":

                NUM_CLASSES = 2
                # Configurations
                lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
                epochs= trial.suggest_int('epochs', 3, 20000)
                hidden_dim = trial.suggest_int('hidden_dim', 4, 128)
                number_of_features = trial.suggest_int('number_of_features', 3, 19)

                self.label_x_selected = self.ranked_features[:number_of_features]

                #TODO get feature selection in hyperparameter tuning

                # get proper amount of features
                x_tune_c = train_for_tuning.loc[train_for_tuning[self.label_t]==0][self.label_x_selected].copy()
                x_tune_t = train_for_tuning.loc[train_for_tuning[self.label_t]==1][self.label_x_selected].copy()

                y_tune_c = train_for_tuning.loc[train_for_tuning[self.label_t]==0][self.label_y].copy()
                y_tune_t = train_for_tuning.loc[train_for_tuning[self.label_t]==1][self.label_y].copy()

                x_valid_c = train_for_valid.loc[train_for_valid[self.label_t]==0][self.label_x_selected].copy()
                x_valid_t = train_for_valid.loc[train_for_valid[self.label_t]==1][self.label_x_selected].copy()

                y_valid_c = train_for_valid.loc[train_for_valid[self.label_t]==0][self.label_y].copy()
                y_valid_t = train_for_valid.loc[train_for_valid[self.label_t]==1][self.label_y].copy()

                # convert df to tensor
                train_for_tuning = to_tensor(train_for_tuning[self.label_x_selected]).to(device)

                x_tune_c = to_tensor(x_tune_c).to(device)
                x_tune_t = to_tensor(x_tune_t).to(device)

                y_tune_c = to_tensor(y_tune_c).to(device)
                y_tune_t = to_tensor(y_tune_t).to(device)

                model = TARnetIMA_CFRClassifier(train_for_tuning.shape[1], 0.01, hidden_dim=hidden_dim, n_classes=1).to(device)

                criterion = nn.BCELoss()#nn.NLLLoss()#nn.CrossEntropyLoss()

                # initialise optimiser
                optimizer = optim.Adam(model.parameters(), lr)

                # Define loss lists
                loss1_lst, loss2_lst, loss3_lst = [], [], []


                for epoch in range(epochs):

                    # Forward pass
                    control_t, control_c = model(x_tune_c)
                    treated_t, treated_c = model(x_tune_t)

                    # Compute total loss and update the model's parameters
                    loss1, loss2 = criterion(treated_t, y_tune_t.reshape(-1,1)), criterion(control_c, y_tune_c.reshape(-1,1))
                    
                    # losses added
                    loss = loss1 + loss2 
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Print the loss every 100 epochs
                    if (epoch + 1) % 1000 == 0:
                        print(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')
                        
                        # y0, y1 = model(train)
                        # ites = y1 - y0

                        # # print intermediate ATE predictions
                        # print(torch.mean(ites))
                    

                x_valid = to_tensor(train_for_valid[self.label_x_selected]).to(device)

                # make predictions
                y_pred_valid_t, y_pred_valid_c = model(x_valid)

                # get actual prediction, convert to numpy for working with pandas
                if torch.cuda.is_available() is True:
                    y_pred_valid_c = y_pred_valid_c.cpu().detach().numpy()
                    y_pred_valid_t = y_pred_valid_t.cpu().detach().numpy()
                if torch.cuda.is_available() is False:
                    y_pred_valid_c = y_pred_valid_c.detach().numpy()
                    y_pred_valid_t = y_pred_valid_t.detach().numpy()

                train_for_valid['y_pred_valid_c'] = y_pred_valid_c
                train_for_valid['y_pred_valid_t'] = y_pred_valid_t

                train_for_valid['observed_prediction'] = np.NaN
                train_for_valid.loc[train_for_valid[self.label_t] == 0, 'observed_prediction'] = train_for_valid['y_pred_valid_c']
                train_for_valid.loc[train_for_valid[self.label_t] == 1, 'observed_prediction'] = train_for_valid['y_pred_valid_t']

                # get auc
                auc = roc_auc_score(train_for_valid[self.label_y], train_for_valid['observed_prediction']) # choose y0 or y1 depending on the observed outcome (choose the verifiable outcome)
            
                # # get brier score loss
                # brier = brier_score_loss(train_for_valid[self.label_y], train_for_valid['observed_prediction'])

                #trial.report(auc, epoch)

                trial.report(auc, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned

            return auc

        study_name = "hyperparameter_tuning_v2"
        storage = f"sqlite:///{ROOT}/logs/{study_name}.db"
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=10)

        trial = study.best_trial

        print(f"AUC: {trial.value}")
        print(f"Best hyperparameters: {trial.params}")
        
        logger.info(f"AUC: {trial.value}")
        logger.info(f"Best hyperparameters: {study.best_params}")

        logger.info('.hyperparameter_tuning() ran succesfully')

        


