import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from src.TARNetCodeBase.metrics import pehe_eval
from src.TARNetCodeBase.empirical_data import ihdp_loader
from src.TARNetCodeBase.doubly_robust import doubly_robust
from src.TARNetCodeBase.TARNet import TARnetICFR
from src.TARNetCodeBase.IMAFCRNet import IMAFCRNet
from src.TARNetCodeBase.TARnetIMA_CFRClassifier import TARnetIMA_CFRClassifier


from utils import ROOT, to_tensor, device

import logging
logger = logging.getLogger('run')


np.random.seed(42)



class Mixin:
    def train_model(self):

        # assign train df (use data without imputation for XGBoost model). special case for x-learner because it uses LogisticRegression for g
        if (self.metalearner_classifier == 'XGB') and (self.algorithm != 'x-learner'):
            train = self.preprocessed_data_scaled.loc[self.preprocessed_data_scaled[self.label_id].isin(self.train_ids)].copy()
        if (self.metalearner_classifier != 'XGB') or (self.algorithm == 'x-learner'):
            train = self.preprocessed_data_scaled_imputed.loc[self.preprocessed_data_scaled_imputed[self.label_id].isin(self.train_ids)].copy()

        # set number of features

        if self.algorithm == 'tarnet-imacfr':
            # number_of_features = 13 # based on feature selection plot
            number_of_features = 11 # based on hyperparameter tuning 28-3-2024 v2. PM: also change config settings below!
        else:
            number_of_features = 13 # based on feature selection plot

        # select top features for features
        self.label_x_selected = self.ranked_features[:number_of_features]

        # for s-learner, add  treatment as variable
        if self.algorithm == 's-learner':
            self.label_x_selected.append(self.label_t)

        # export to object for external validation
        with open(ROOT + f'\\saved_models\\{self.name}_label_x_selected.pkl', 'wb') as f:
            pickle.dump(self.label_x_selected, f)
        f.close()

        # specify specific training sets splitted based on treatment
        x_train_c = train.loc[train[self.label_t]==0][self.label_x_selected].copy()
        x_train_t = train.loc[train[self.label_t]==1][self.label_x_selected].copy()

        y_train_c = train.loc[train[self.label_t]==0][self.label_y].copy()
        y_train_t = train.loc[train[self.label_t]==1][self.label_y].copy()

        if self.algorithm == "tarnet":

            # # convert to tensor
            # x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c = df_to_tensor(
            #     x_train_t=self.x_train_t, 
            #     x_train_c=self.x_train_c, 
            #     y_train_t=self.y_train_t, 
            #     y_train_c=self.y_train_c, 
            #     x_test_t=self.x_test_t, 
            #     x_test_c=self.x_test_c, 
            #     y_test_t=self.y_test_t, 
            #     y_test_c=self.y_test_c
            #     )

            # get tau_est
            # tau_est = get_tau_est(data=data, X=X, T=T, Y=Y)

            # input in TARNet # TODO split predictions from actual modeling training
            model, head_loss, m, tau_est_model = run_tarnet(x_train_t=x_train_t, x_train_c=x_train_c, y_train_t=y_train_t, y_train_c=y_train_c)

            self.tau_est = tau_est_model # tau est
            self.model = model
            self.head_loss = head_loss
            self.m = m
            # self.y_pred_test_t = y_pred_test_t
            # self.y_pred_test_c = y_pred_test_c

            # export model -> no longer necessary, split for code below
            #export_model(model)

            with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            f.close()

        if self.algorithm == "imaf":
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

            # input in TARNet # TODO split predictions from actual modeling training
            model, head_loss, m, tau_est_model = run_imaf(x_train_t=x_train_t, x_train_c=x_train_c, y_train_t=y_train_t, y_train_c=y_train_c)       

            self.tau_est = tau_est_model # tau est
            self.model = model
            self.head_loss = head_loss
            self.m = m

        if self.algorithm == "tarnet-imacfr":
            NUM_CLASSES = 2
    
            lr = 5.31782611449544e-06
            epochs = 11197
            hidden_dim = 108
            

            # convert df to tensor
            train = to_tensor(train[self.label_x_selected]).to(device)

            x_train_c = to_tensor(x_train_c).to(device)
            x_train_t = to_tensor(x_train_t).to(device)

            y_train_c = to_tensor(y_train_c).to(device)
            y_train_t = to_tensor(y_train_t).to(device)

            model = TARnetIMA_CFRClassifier(train.shape[1], 0.01, hidden_dim = hidden_dim, n_classes = 1).to(device)

            criterion = nn.BCELoss()#nn.NLLLoss()#nn.CrossEntropyLoss()

            # initialise optimiser
            optimizer = optim.Adam(model.parameters(), lr)

            # Define loss lists
            loss1_lst, loss2_lst, loss3_lst = [], [], []

            for epoch in range(epochs):

                # Forward pass TODO check if T needs to be in the model ? How is each treatment assigned ?
                control_t, control_c = model(x_train_c)
                treated_t, treated_c = model(x_train_t)

                # Compute total loss and update the model's parameters
                loss1, loss2 = criterion(treated_t, y_train_t.reshape(-1,1)), criterion(control_c, y_train_c.reshape(-1,1))
                
                # losses added
                loss = loss1 + loss2 
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print the loss every 100 epochs
                if (epoch + 1) % 1000 == 0:
                    print(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')
                    
                    y0, y1 = model(train)
                    ites = y1 - y0

                    # print intermediate ATE predictions
                    print(torch.mean(ites))

            # save model
            with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            f.close()

            # save to attribute
            self.model = model

        if self.algorithm == "s-learner":
            model = self.init_clf_for_metalearner()

            x_train = train[self.label_x_selected].copy()
            y_train = train[self.label_y].copy()

            model.fit(x_train, y_train)

            self.model = model

        if (self.algorithm == "t-learner") or (self.algorithm == "x-learner"):

            m0, m1 = self.init_clf_for_metalearner() # code below, written asa function to use for second stage in x-learner

            m0.fit(x_train_c, y_train_c)
            m1.fit(x_train_t, y_train_t)

            if self.algorithm == "x-learner":
                # make propensity score model
                g = LogisticRegression(solver="lbfgs", penalty='none')
                
                # # create train[X] and train[T]
                # train_t = self.x_train_t
                # train_t[self.label_t] = 1
                # train_t[self.label_y] = self.y_train_t.copy()
                # self.train_t = train_t

                # train_c = self.x_train_c
                # train_c[self.label_t] = 0
                # train_c[self.label_y] = self.y_train_c.copy()
                # self.train_c = train_c

                # train = pd.concat([train_t,train_c])
                # self.train = train

                # fit propensity score model
                g.fit(train[self.label_x_selected], train[self.label_t])
                self.g = g

                with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_{self.metalearner_classifier}_g.pkl', 'wb') as f:
                    pickle.dump(g, f)
                f.close()

                # TODO impute treatment effect
                d_train = np.where(
                                    train[self.label_t]==0,
                                    m1.predict(train[self.label_x_selected]) - train[self.label_y],
                                    train[self.label_y] - m0.predict(train[self.label_x_selected])
                                )
                
                # conversion for XGBoost format
                le = LabelEncoder()
                d_train = le.fit_transform(d_train)
                
                # second stage
                mx0, mx1 = self.init_clf_for_metalearner()

                mx0.fit(train.query(f"{self.label_t}==0")[self.label_x_selected], d_train[train[self.label_t]==0])
                mx1.fit(train.query(f"{self.label_t}==1")[self.label_x_selected], d_train[train[self.label_t]==1])
                
                self.mx0 = mx0
                with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_{self.metalearner_classifier}_mx0_model.pkl', 'wb') as f:
                    pickle.dump(mx0, f)
                f.close()

                self.mx1 = mx1
                with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_{self.metalearner_classifier}_mx1_model.pkl', 'wb') as f:
                    pickle.dump(mx1, f)
                f.close()

            # attribute to model
            self.m0 = m0
            self.m1 = m1

        if self.algorithm == "psm":

             # Step 1: Fit propensity score model
            x_train = train[self.label_x_selected].copy()
            t_train = train[self.label_t].copy()

            ps_model = LogisticRegression(C=1e6).fit(x_train, t_train)

            # save model
            with open(ROOT + f"\\saved_models\\{self.source}_{self.algorithm}_propensity_score_model.pkl", "wb") as f:
                pickle.dump(ps_model, f)

            propensity_scores = ps_model.predict_proba(x_train)[:,1]

            # store as attribute
            self.ps_model = ps_model

            train_ps = train.copy()
            train_ps['propensity_score'] = propensity_scores

            # Step 2: Split into treated and control groups
            treated = train_ps[train_ps[self.label_t]==1]
            control = train_ps[train_ps[self.label_t]==0]

            # Step 3: Nearest neighbor matching (1:1) to match based on propensity scores
            nearn = NearestNeighbors(n_neighbors=1)
            nearn.fit(control[['propensity_score']])
            distances, indices = nearn.kneighbors(treated[['propensity_score']])

            matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
            treated = treated.reset_index(drop=True)

            # Step 4: Compute ITEs with two models that estimate ites
            m0, m1 = self.init_clf_for_metalearner() # code below

            m0.fit(matched_control[self.label_x_selected], matched_control[self.label_y])
            m1.fit(treated[self.label_x_selected], treated[self.label_y])

            self.m0 = m0
            self.m1 = m1

            with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_m0_model.pkl', 'wb') as f:
                pickle.dump(m0, f)
            f.close()

            with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_m1_model.pkl', 'wb') as f:
                pickle.dump(m1, f)
            f.close()

        if self.metalearner_classifier is not None:

            if self.m0 is not None: # for t-learner, x-learner
                with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_{self.metalearner_classifier}_m0_model.pkl', 'wb') as f:
                    pickle.dump(m0, f)
                f.close()

            if self.m1 is not None: # for t-learner, x-learner
                with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_{self.metalearner_classifier}_m1_model.pkl', 'wb') as f:
                    pickle.dump(m1, f)

                f.close()

            if self.model is not None: # for s-learner
                with open(ROOT + f'\\saved_models\\{self.source}_{self.algorithm}_{self.metalearner_classifier}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)

                f.close()

        logger.info('.train_model() ran succesfully')

    def init_clf_for_metalearner(self):

        if self.algorithm == "s-learner":

            # LR
            if self.metalearner_classifier == "LR":
            
                model = LogisticRegression(penalty='none', max_iter=10000, random_state=self.random_state)

            # XGB
            if self.metalearner_classifier == "XGB":
                # Train a model using the scikit-learn API
                model = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=3, enable_categorical=True)

            # MLP
            if self.metalearner_classifier == "MLP":
                model = MLPClassifier(random_state=self.random_state, max_iter=300)
            
            return model
            
        if (self.algorithm == "t-learner") or (self.algorithm == "x-learner"):

            if self.metalearner_classifier == "LR":
                # LR
                m0 = LogisticRegression(penalty='none', max_iter=10000, random_state=self.random_state)
                m1 = LogisticRegression(penalty='none', max_iter=10000, random_state=self.random_state)

            if self.metalearner_classifier == "XGB":
                # Train a model using the scikit-learn API
                m0 = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=3, enable_categorical=True)
                m1 = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=3, enable_categorical=True)
                
            if self.metalearner_classifier == "MLP":
                m0 = MLPClassifier(random_state=self.random_state, max_iter=300)
                m1 = MLPClassifier(random_state=self.random_state, max_iter=300)

            return m0, m1

        else:
            return print('no valid algorithm')

    

def label_data(data):
    
    # load data
    X = data.drop(columns=['admissionid', '28daymortality', 'high_dose_steroids']).columns.tolist()
    Y = '28daymortality'
    T = 'high_dose_steroids'

    return X, Y, T

def df_to_tensor(x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c):
    to_tensor = lambda x: torch.tensor(np.array(x), dtype = torch.float32).to(device)
 
    x_train_t = to_tensor(np.array(x_train_t, dtype='float32'))
    x_train_c = to_tensor(np.array(x_train_c, dtype='float32'))

    y_train_t = to_tensor(np.array(y_train_t, dtype='float32'))
    y_train_c = to_tensor(np.array(y_train_c, dtype='float32'))

    x_test_t = to_tensor(np.array(x_test_t, dtype='float32'))
    x_test_c = to_tensor(np.array(x_test_c, dtype='float32'))

    y_test_t = to_tensor(np.array(y_test_t, dtype='float32'))
    y_test_c = to_tensor(np.array(y_test_c, dtype='float32'))
    
    return x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c

def get_tau_est(data, X, Y, T):
    # Estimate the average treatment effect (group level)
    tau_est = doubly_robust(
        data, 
        X=X, 
        T=T, 
        Y=Y)

    tau_est = torch.tensor(tau_est, dtype = torch.float32).to(device)

    return tau_est

def run_tarnet(x_train_t, x_train_c, y_train_t, y_train_c):

    # TODO loss3, tau_est, en op test set loss berekenen

    # apply TARnet

    # # Estimate the average treatment effect (group level)
    # tau_est = doubly_robust(
    #     df, 
    #     X=["Covariate_1", "Covariate_2", "Covariate_3"], 
    #     T="Treatment", 
    #     Y="Outcome")

    # tau_est = torch.tensor(tau_est, dtype = torch.float32).to(device)

    # Configurations for TARNet
    lr = 0.0001
    epochs = 6000 
    gamma = 2

    model = TARnetICFR(x_train_t.shape[1], 0.01).to(device)

    # Define the loss functions
    #head_loss = nn.L1Loss() 
    head_loss = nn.BCELoss()
    m = nn.Sigmoid()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr)

    # Define loss lists
    loss1_lst, loss2_lst, loss3_lst = [], [], []

    for epoch in range(epochs):
        # Forward pass
        output_t, output_c, tau_est_model = model.forward(x_train_t, x_train_c)

        # Compute total loss and update the model's parameters
        loss1, loss2 = head_loss(m(output_t), y_train_t.view(-1, 1)), head_loss(m(output_c), y_train_c.view(-1, 1))#, head_loss(tau_est, tau_est_model)
     
        # losses added
        loss = loss1 + loss2 #+ (gamma * loss3)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1_lst.append(loss1.item())
        loss2_lst.append(loss2.item())
        #loss3_lst.append(loss3.item())

        # Print the loss every 10000 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')
    
    return model, head_loss, m, tau_est_model

def run_imaf(x_train_t, x_train_c, y_train_t, y_train_c):
    
    # TODO loss3, tau_est, en op test set loss berekenen

    # apply TARnet

    # # Estimate the average treatment effect (group level)
    # tau_est = doubly_robust(
    #     df, 
    #     X=["Covariate_1", "Covariate_2", "Covariate_3"], 
    #     T="Treatment", 
    #     Y="Outcome")

    # tau_est = torch.tensor(tau_est, dtype = torch.float32).to(device)

    # Configurations for TARNet
    lr = 0.0001
    epochs = 6000 
    gamma = 2
    model = IMAFCRNet(x_train_t.shape[1], 64)

    # Define the loss functions
    #head_loss = nn.L1Loss() 
    head_loss = nn.BCELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr)

    # Define loss lists
    loss1_lst, loss2_lst, loss3_lst = [], [], []

    for epoch in range(epochs):
        # Forward pass
        output_t = model(x_train_t) # 1 positional argument, see louk script, [y0, y1]

        output_c = model(x_train_c) # [y0, y1]

        # Compute total loss and update the model's parameters
        loss1, loss2 = head_loss(output_t, y_train_t.view(-1, 1)), head_loss(output_c, y_train_c.view(-1, 1))#, head_loss(tau_est, tau_est_model)
     
        # losses added
        loss = loss1 + loss2 #+ (gamma * loss3)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1_lst.append(loss1.item())
        loss2_lst.append(loss2.item())
        #loss3_lst.append(loss3.item())

        # Print the loss every 10000 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')
    
    return model, head_loss

def predictions(model, head_loss, m, x_test_t, x_test_c):

    y_pred_test_t, y_pred_test_c, _ = model(x_test_t, x_test_c)
    loss1_test, loss2_test = head_loss(m(y_pred_test_t), y_test_t.view(-1, 1)), head_loss(m(y_pred_test_c), y_test_c.view(-1, 1))

    print("Loss1_test: ", loss1_test.item())
    print("Loss2_test: ", loss2_test.item())

    # confusion matrix variables
    y_pred_test_t, y_pred_test_c, _ = model(x_test_t, x_test_c)
    y_pred_test_t = (m(y_pred_test_t) > 0.5).cpu().detach().numpy().astype(int).flatten()
    y_pred_test_c = (m(y_pred_test_c) > 0.5).cpu().detach().numpy().astype(int).flatten()

    y_test_t = y_test_t.cpu().detach().numpy().astype(int).flatten()
    y_test_c = y_test_c.cpu().detach().numpy().astype(int).flatten()

    # Confusion matrix for treatment group
    cm_t = confusion_matrix(y_test_t, y_pred_test_t)
    print("Confusion Matrix - Treatment Group:")
    print(cm_t)

    # Confusion matrix for control group
    cm_c = confusion_matrix(y_test_c, y_pred_test_c)
    print("Confusion Matrix - Control Group:")
    print(cm_c)

    # Confusion matrix total
    cm_total = cm_t + cm_c
    print("Confusion Matrix - Total:")
    print(cm_total)


    return y_pred_test_t, y_pred_test_c 

def export_model(model):
    with open(ROOT + '\\saved_csvs\\tarnet_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    f.close()