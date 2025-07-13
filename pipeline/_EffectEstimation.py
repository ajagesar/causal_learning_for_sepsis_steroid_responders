from utils import ROOT, to_tensor
from _Modeling import df_to_tensor, device
import torch
import numpy as np
import pandas as pd
import ot
from sklearn.neighbors import NearestNeighbors

import logging
logger = logging.getLogger('run')
class Mixin:
    def ite_estimation(self):
        # ite_t = predicted outcome under treatment minus prediction outcome under control

        # TODO convert this to a function
        if (self.metalearner_classifier == 'XGB') and (self.algorithm != 'x-learner'):
            data = self.preprocessed_data_scaled.copy()
        if (self.metalearner_classifier != 'XGB') or (self.algorithm == 'x-learner'):
            data = self.preprocessed_data_scaled_imputed.copy()

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

            # df to tensor
            #to_tensor = lambda x: torch.tensor(np.array(x), dtype = torch.float32).to(device)
            x_t = to_tensor(np.array(self.x_t, dtype='float32')).to(device)
            x_c = to_tensor(np.array(self.x_c, dtype='float32')).to(device)
            y_t = to_tensor(np.array(self.y_t, dtype="float32")).to(device)
            y_c = to_tensor(np.array(self.y_c, dtype="float32")).to(device)

            # effect estimation on total set
            y1, y0, tau_est = self.model.forward(treated=x_t, control=x_c)

            self.ite_t = y1
            self.ite_c = y0
            self.ite = torch.cat([self.ite_t, self.ite_c], dim=0)

        if self.algorithm == "tarnet-imacfr":
            
            x = to_tensor(data[self.label_x_selected])

            y_pred_t, y_pred_c = self.model(x)

            if torch.cuda.is_available() is True:
                y_pred_c = y_pred_c.cpu().detach().numpy()
                y_pred_t = y_pred_t.cpu().detach().numpy()
            if torch.cuda.is_available() is False:
                y_pred_c = y_pred_c.detach().numpy()
                y_pred_t = y_pred_t.detach().numpy()

            data['y_pred_c'] = y_pred_c
            data['y_pred_t'] = y_pred_t

            data['ite'] = data['y_pred_t'] - data['y_pred_c']

            self.ite = data['ite'].to_numpy()

            print("Calculating Wasserstein")

            ## create df from latent representation #TODO fix below -> get Wasserstein; PM add regularization for TARNet -> CFR
            latent_df_control = pd.DataFrame(self.model.phi_control.detach().numpy())
            latent_df_treated = pd.DataFrame(self.model.phi_treated.detach().numpy())

            # compute Wasserstein
            self.wasserstein_before_matching = compute_wasserstein_ot(
                arr1=data.loc[data[self.label_t]==0][self.label_x].to_numpy(),
                arr2=data.loc[data[self.label_t]==1][self.label_x].to_numpy()
                )

            self.wasserstein_after_matching = compute_wasserstein_ot(
                arr1=latent_df_control.to_numpy(),
                arr2=latent_df_treated.to_numpy()
            )


        if self.algorithm == "t-learner":
            # TODO fix with changed imputation code -> 2x imputing ?

            # #TODO change quickfix below
            # data['mechanical_ventilation_bool'] = data['mechanical_ventilation_bool'].astype(bool)

            # data = data.fillna(data.median())

            self.ite = self.m1.predict_proba(data[self.label_x_selected])[:,1] - self.m0.predict_proba(data[self.label_x_selected])[:,1]


        if self.algorithm == "x-learner":

            self.ite = (
                        self.ps_predict(data[self.label_x_selected],1) * self.mx0.predict(data[self.label_x_selected]) + 
                        self.ps_predict(data[self.label_x_selected],0) * self.mx1.predict(data[self.label_x_selected])
                        )
            
        if self.algorithm == "s-learner":
            
            self.label_x_selected.remove(self.label_t) # remove treatment column from features

            data_treatment_given = data[self.label_x_selected].copy() # create treatment patients
            data_treatment_given[self.label_t] = 1

            data_treatment_not_given = data[self.label_x_selected].copy() # create control patients
            data_treatment_not_given[self.label_t] = 0

            # is label_t in self.label_x_selected ? or do i have to add it now before plugging it in the model ?
            self.label_x_selected.append(self.label_t) # readd to features column            

            self.ite = self.model.predict_proba(data_treatment_given[self.label_x_selected])[:,1] - self.model.predict_proba(data_treatment_not_given[self.label_x_selected])[:,1]

        if self.algorithm == "psm":

            propensity_scores = self.ps_model.predict_proba(data[self.label_x_selected])[:,1]
            data['propensity_score'] = propensity_scores

            # Split into treated and control groups
            treated = data[data[self.label_t]==1].copy()
            control = data[data[self.label_t]==0].copy()

            # Nearest neighbor matching (1:1) to match based on propensity scores
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(control[['propensity_score']])
            distances, indices = nn.kneighbors(treated[['propensity_score']])

            matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
            treated = treated.reset_index(drop=True)
            matched_data = pd.concat([treated, matched_control])

            self.ite = self.m1.predict_proba(treated[self.label_x_selected])[:,1] - self.m0.predict_proba(matched_control[self.label_x_selected])[:,1]

            # compute Wasserstein
            print("Calculating Wasserstein")

            self.wasserstein_before_matching = compute_wasserstein_ot(
                arr1=data.loc[data[self.label_t]==0][self.label_x].to_numpy(),
                arr2=data.loc[data[self.label_t]==1][self.label_x].to_numpy()
            )

            self.wasserstein_after_matching = compute_wasserstein_ot(
                arr1=treated[self.label_x].to_numpy(),
                arr2=matched_control[self.label_x].to_numpy()
            )

        logger.info('.ite_estimation() ran succesfully')

        if (self.wasserstein_before_matching != None) & (self.wasserstein_after_matching != None):

            with open(ROOT + f"\\paper\\output\\{self.name}_wasserstein.txt", "w") as file:
                print(f"Wasserstein before modeling: {self.wasserstein_before_matching}", file=file)
                print(f"Wasserstein after modeling: {self.wasserstein_after_matching}", file=file)

    def ps_predict(self, df, t):
        return self.g.predict_proba(df[self.label_x_selected])[:, t]
    
def compute_wasserstein_ot(arr1, arr2):

    # Uniform weights for samples -> makes each sample equally important
    a = np.ones(arr1.shape[0]) / arr1.shape[0]
    b = np.ones(arr2.shape[0]) / arr2.shape[0]

    # Cost matrix (e.g., Euclidean distance between samples)
    m = ot.dist(arr1, arr2)

    # Wasserstein distance using optimal transport
    emd_distance = ot.emd2(a, b, m, numItermax=1000000)

    return emd_distance