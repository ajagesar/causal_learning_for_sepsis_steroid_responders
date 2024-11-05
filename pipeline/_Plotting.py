from utils import ROOT
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('run')

class Mixin:
    def create_plot(self):

        if self.metalearner_classifier == 'XGB':
            test = self.preprocessed_data_scaled.loc[self.preprocessed_data_scaled[self.label_id].isin(self.test_ids)].copy()
        if self.metalearner_classifier != 'XGB':
            test = self.preprocessed_data_scaled_imputed.loc[self.preprocessed_data_scaled_imputed[self.label_id].isin(self.test_ids)].copy()

        y_test_c = test.loc[test[self.label_t]==0][self.label_y].copy()
        y_test_t = test.loc[test[self.label_t]==1][self.label_y].copy()

        if self.algorithm == "tarnet":

            # convert to numpy array
            # y_test_t = y_test_t.to_numpy()
            # y_test_c = y_test_c.to_numpy()

            # y_pred_test_t = self.y_pred_test_t.cpu().detach().numpy().flatten()
            # y_pred_test_c = self.y_pred_test_c.cpu().detach().numpy().flatten()

            # combine treatment and control groups for auc curve
            y_test = np.concatenate((y_test_t, y_test_c), axis=None)
            y_pred_test = np.concatenate((y_pred_test_t, y_pred_test_c), axis=None)

        if self.algorithm == "tarnet-imacfr":
            y_test = test[self.label_y].to_numpy()
            y_pred_test = self.y_pred_test


        if (self.algorithm == "t-learner") or (self.algorithm == "x-learner"):
            # call predictions
            y_pred_test_t = self.y_pred_test_t
            y_pred_test_c = self.y_pred_test_c     

            # concat to y_test
            y_test = pd.concat([y_test_t, y_test_c])
            y_pred_test = np.concatenate([y_pred_test_t[:,1], y_pred_test_c[:,1]], axis=None)

        if (self.algorithm == "s-learner"):
            # call prediction
            y_test = test[self.label_y].copy()
            y_pred_test = self.y_pred_test

        # roc curve
        fpr, tpr, threshold = roc_curve(y_test, y_pred_test)
        roc_auc = auc(fpr, tpr)

        plt.figure() # clear plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' %roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(ROOT + f'\\saved_csvs\\{self.name}_roc_curve.png')


        # calibration curve
        true_pos, pred_pos = calibration_curve(y_test, y_pred_test, n_bins=10)

        plt.figure() # clear plt
        plt.title('Calibration plot')
        plt.xlabel('Predicted probability')
        plt.ylabel('True probability')

        plt.plot(pred_pos, true_pos, marker='o', label=self.name)
        plt.plot([0, 1], [0, 1], linestyle='--') # perfect calibration
        plt.legend(loc='lower right')
        plt.savefig(ROOT + f'\\saved_csvs\\{self.name}_calibration_curve.png')

        # save plotting parameters
        self.plt_fpr = fpr
        self.plt_tpr = tpr
        self.plt_roc_auc = roc_auc
        self.plt_true_pos = true_pos
        self.plt_pred_pos = pred_pos

    logger.info('.create_plot() ran succesfully')

        # visualize ites

    # def show_roc_curve(self):
    #     plt.figure(self.roc_curve)
    #     plt.show()