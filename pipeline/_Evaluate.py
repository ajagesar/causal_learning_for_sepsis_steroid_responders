from utils import ROOT, to_tensor, device
#from torcheval.metrics.aggregation.auc import AUC
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
from _Modeling import df_to_tensor
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger('run')

class Mixin:
    def evaluate(self):

        if (self.metalearner_classifier == 'XGB') and (self.algorithm != 'x-learner'):
            test = self.preprocessed_data_scaled.loc[self.preprocessed_data_scaled[self.label_id].isin(self.test_ids)].copy()
        if (self.metalearner_classifier != 'XGB') or (self.algorithm == 'x-learner'):
            test = self.preprocessed_data_scaled_imputed.loc[self.preprocessed_data_scaled_imputed[self.label_id].isin(self.test_ids)].copy()

        x_test_c = test.loc[test[self.label_t]==0][self.label_x_selected].copy()
        x_test_t = test.loc[test[self.label_t]==1][self.label_x_selected].copy()

        y_test_c = test.loc[test[self.label_t]==0][self.label_y].copy()
        y_test_t = test.loc[test[self.label_t]==1][self.label_y].copy()

        if self.algorithm == "tarnet":

            # convert to tensor
            x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c = df_to_tensor(
                x_train_t=self.x_train_t, 
                x_train_c=self.x_train_c, 
                y_train_t=self.y_train_t, 
                y_train_c=self.y_train_c, 
                x_test_t=self.x_test_t, 
                x_test_c=self.x_test_c, 
                y_test_t=self.y_test_t, 
                y_test_c=self.y_test_c)

            # predictions TODO find out if epoch > 6000 makes a difference
            y_pred_test_t, y_pred_test_c, _ = self.model(x_test_t, x_test_c)

            self.loss1_test, self.loss2_test = self.head_loss(self.m(y_pred_test_t), y_test_t.view(-1, 1)), self.head_loss(self.m(y_pred_test_c), y_test_c.view(-1, 1))

            # confusion matrix variables
            y_pred_test_t_cm = (self.m(y_pred_test_t) > 0.5).cpu().detach().numpy().astype(int).flatten()
            y_pred_test_c_cm = (self.m(y_pred_test_c) > 0.5).cpu().detach().numpy().astype(int).flatten()

            y_test_t_cm = y_test_t.cpu().detach().numpy().astype(int).flatten()
            y_test_c_cm = y_test_c.cpu().detach().numpy().astype(int).flatten()

            # Confusion matrix for treatment group
            cm_t = confusion_matrix(y_test_t_cm, y_pred_test_t_cm)

            # Confusion matrix for control group
            cm_c = confusion_matrix(y_test_c_cm, y_pred_test_c_cm)

            # Confusion matrix total
            self.confusion_matrix = cm_t + cm_c

            # get auc
            y_test = torch.cat((y_test_t, y_test_c), dim=0)
            y_pred_test = torch.cat((y_pred_test_t, y_pred_test_c), dim=0)
            y_pred_test = y_pred_test.squeeze()

            y_test_np = y_test.numpy()
            y_pred_test_np = y_pred_test.detach().numpy()

            auc = roc_auc_score(y_test_np, y_pred_test_np)

            # get brier score loss
            brier = brier_score_loss(y_test_np, y_pred_test_np)

        if self.algorithm == "tarnet-imacfr":
            
            x_test = to_tensor(test[self.label_x_selected])

            # make predictions
            y_pred_test_t, y_pred_test_c = self.model(x_test)

            # get actual prediction, convert to numpy for working with pandas
            if torch.cuda.is_available() is True:
                y_pred_test_c = y_pred_test_c.cpu().detach().numpy()
                y_pred_test_t = y_pred_test_t.cpu().detach().numpy()
            if torch.cuda.is_available() is False:
                y_pred_test_c = y_pred_test_c.detach().numpy()
                y_pred_test_t = y_pred_test_t.detach().numpy()

            test['y_pred_test_c'] = y_pred_test_c
            test['y_pred_test_t'] = y_pred_test_t

            test['observed_prediction'] = np.NaN
            test.loc[test[self.label_t] == 0, 'observed_prediction'] = test['y_pred_test_c']
            test.loc[test[self.label_t] == 1, 'observed_prediction'] = test['y_pred_test_t']

            self.y_pred_test = test['observed_prediction'].to_numpy()
 
            # get auc
            auc = roc_auc_score(test[self.label_y], test['observed_prediction']) # choose y0 or y1 depending on the observed outcome (choose the verifiable outcome)
            
            # get brier score loss
            brier = brier_score_loss(test[self.label_y], test['observed_prediction'])

        # TODO: is this okay ?
        if (self.algorithm == "t-learner") or (self.algorithm == "x-learner"):
            # make predictions
            y_pred_test_t = self.m1.predict_proba(x_test_t)
            y_pred_test_c = self.m0.predict_proba(x_test_c)

            # confusion matrix for treatment group
            y_pred_test_t_cm = self.m1.predict(x_test_t) # get binary prediction
            cm_t = confusion_matrix(y_test_t, y_pred_test_t_cm)

            # confusion matrix for control group
            y_pred_test_c_cm = self.m0.predict(x_test_c)
            cm_c = confusion_matrix(y_test_c, y_pred_test_c_cm)
            
            # Confusion matrix total
            self.confusion_matrix = cm_t + cm_c

            # auc
            y_test = pd.concat([y_test_t, y_test_c])
            y_pred_test = np.concatenate([y_pred_test_t[:,1], y_pred_test_c[:,1]], axis=None)
            # PM code to only get one of the two y_pred_test rows

            auc = roc_auc_score(y_test, y_pred_test)

            # brier score loss
            brier = brier_score_loss(y_test, y_pred_test)

        if self.algorithm == 's-learner':

            x_test = test[self.label_x_selected].copy()
            y_test = test[self.label_y].copy()

            # make predictions
            y_pred_test = self.model.predict_proba(x_test)[:,1]

            # confusion matrix
            y_pred_test_binary = self.model.predict(x_test)
            cm = confusion_matrix(y_test, y_pred_test_binary)
            self.confusion_matrix = cm

            # auc
            auc = roc_auc_score(y_test, y_pred_test)

            # brier score loss
            brier = brier_score_loss(y_test, y_pred_test)
                    
        # set attributes
        if self.algorithm == 's-learner':
            self.y_pred_test = y_pred_test

        else:
            self.y_pred_test_t = y_pred_test_t
            self.y_pred_test_c = y_pred_test_c

        self.auc = auc
        self.brier_score_loss = brier

        logger.info('.evaluate() ran succesfully')

        # TODO use .forward for tarnet

    