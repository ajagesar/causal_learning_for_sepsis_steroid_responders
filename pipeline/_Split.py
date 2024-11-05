import pandas as pd
import numpy as np
from utils import ROOT
import pickle

import logging
logger = logging.getLogger('run')

class Mixin:
    def split(self, random_state=42):
        # read data
        data = pd.read_csv(ROOT + f'\\saved_csvs\\{self.source}_preprocessed_data.csv', index_col=False)

        # create labels for covariates (X), outcome (Y), intervention (T) and id
        X, Y, T = label_data(data=data)
        
        self.label_x = X
        self.label_y = Y
        self.label_t = T

        label_id = self.get_id_label()
        self.label_id = label_id

        # get treated and control groups ids
        self.treated_ids = data.loc[data[T]==1][label_id].to_list()
        self.control_ids = data.loc[data[T]==0][label_id].to_list()

        # split data in train/test set
        train_ids, test_ids = self.split_data()

        self.train_ids = train_ids
        self.test_ids = test_ids

        # save train_ids for paper
        with open(ROOT + '\\paper\\output\\train_ids.pkl', 'wb') as f:
            pickle.dump(train_ids, f)
        f.close()

        # self.x_t_unscaled = data.loc[data[T]==1][X].copy() # mentioning unscaled for interpretation later
        # self.x_c_unscaled = data.loc[data[T]==0][X].copy() # mentioning unscaled for interpretation later
        # self.y_t = data.loc[data[T]==1][Y].copy()
        # self.y_c = data.loc[data[T]==0][Y].copy()

        # split the data in train/test and covariates/outcome based on treatment
        #x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c = split_data(data, X=X, Y=Y, T=T, random_state=random_state)

        # self.x_train_t = x_train_t
        # self.x_train_c = x_train_c
        # self.y_train_t = y_train_t
        # self.y_train_c = y_train_c

        # self.x_test_t = x_test_t
        # self.x_test_c = x_test_c
        # self.y_test_t = y_test_t
        # self.y_test_c = y_test_c

        # self.x_train = pd.concat([x_train_t, x_train_c])
        # self.y_train = pd.concat([y_train_t, y_train_c])

        # self.x_test = pd.concat([x_test_t, x_test_c])
        # self.y_test = pd.concat([y_test_t, y_test_c])

        # export data
        #export_data(self.x_t_unscaled, self.x_c_unscaled, self.y_t, self.y_c, x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c)
        #export_labels(X, Y, T)

        #print("Preprocessed data has been split and saved as csvs. It is also attached as an attribute: .x_train_t, .x_train_c, .y_train_t, .y_train_c, .x_test_t, .x_test_c, .y_test_t, .y_test_c")

        logger.info('.split() ran succesfully')

    def get_id_label(self):
        label_id = 'admissionid'
        return label_id
        
    def split_data(self, frac=0.8):
        train = self.preprocessed_data.sample(frac=frac, random_state=self.random_state)
        test = self.preprocessed_data.drop(train.index)

        train_ids = train[self.label_id].to_list()
        test_ids = test[self.label_id].to_list()

        return train_ids, test_ids


def label_data(data):
    
    # load data
    X = data.drop(columns=['admissionid', '28daymortality', 'high_dose_steroids']).columns.tolist()
    Y = '28daymortality'
    T = 'high_dose_steroids'

    return X, Y, T




# def split_data(data, X, Y, T, random_state):
    
#     # split in train and test set
#     train = data.sample(frac=0.8, random_state=random_state)
#     test = data.drop(train.index)

#     # set x_t, x_c, y_t, y_c & x_test_t, x_test_c, y_test_t, y_test_c
#     x_train_t = train.loc[train[T]==1][X].copy()
#     x_train_c = train.loc[train[T]==0][X].copy()

#     y_train_t = train.loc[train[T]==1][Y].copy()
#     y_train_c = train.loc[train[T]==0][Y].copy()

#     x_test_t = test.loc[test[T]==1][X].copy()
#     x_test_c = test.loc[test[T]==0][X].copy()

#     y_test_t = test.loc[test[T]==1][Y].copy()
#     y_test_c = test.loc[test[T]==0][Y].copy()

#     return x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c

def export_data(x_t_unscaled, x_c_unscaled, y_t, y_c, x_train_t, x_train_c, y_train_t, y_train_c, x_test_t, x_test_c, y_test_t, y_test_c): # TODO change below to include mimic name with f-string (have to make it a method from the class to pass self)
    x_t_unscaled.to_csv(ROOT + "\\saved_csvs\\x_t_unscaled.csv")
    x_c_unscaled.to_csv(ROOT + "\\saved_csvs\\x_c_unscaled.csv")
    y_t.to_csv(ROOT + "\\saved_csvs\\y_t.csv")
    y_c.to_csv(ROOT + "\\saved_csvs\\y_c.csv")

    x_train_t.to_csv(ROOT + "\\saved_csvs\\x_train_t.csv")
    x_train_c.to_csv(ROOT + "\\saved_csvs\\x_train_c.csv")
    y_train_t.to_csv(ROOT + "\\saved_csvs\\y_train_t.csv")
    y_train_c.to_csv(ROOT + "\\saved_csvs\\y_train_c.csv")

    x_test_t.to_csv(ROOT + "\\saved_csvs\\x_test_t.csv")
    x_test_c.to_csv(ROOT + "\\saved_csvs\\x_test_c.csv")
    y_test_t.to_csv(ROOT + "\\saved_csvs\\y_test_t.csv")
    y_test_c.to_csv(ROOT + "\\saved_csvs\\y_test_c.csv")

def export_labels(X, Y, T):
    pass # TODO export labels