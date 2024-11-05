import pandas as pd
from utils import ROOT

class Mixin:
    def fake_split(self):
        # make a fake split so the code dependencies don't break, but the entire data set is used for evaluation

        # read data
        data = pd.read_csv(ROOT + f'\\saved_csvs\\{self.source}_preprocessed_data.csv', index_col=False)

        # create labels for covariates (X), outcome (Y) and intervention (T)
        X, Y, T = label_data(data=data)

        self.label_x = X
        self.label_y = Y
        self.label_t = T

        label_id = self.get_id_label()
        self.label_id = label_id

        # get treated and control group ids
        self.treated_ids = data.loc[data[T]==1][label_id].to_list()
        self.control_ids = data.loc[data[T]==0][label_id].to_list()

        # fake split: empty train ids and everything to test_ids
        train_ids = []
        test_ids = data[label_id].to_list()

        self.train_ids = train_ids
        self.test_ids = test_ids

        # self.x_t = data.loc[data[T]==1][X].copy()
        # self.x_c = data.loc[data[T]==0][X].copy()
        # self.y_t = data.loc[data[T]==1][Y].copy()
        # self.y_c = data.loc[data[T]==0][Y].copy()

        # self.x_train_t = self.x_t
        # self.x_train_c = self.x_c
        # self.y_train_t = self.y_t
        # self.y_train_c = self.y_c

        # self.x_test_t = self.x_t
        # self.x_test_c = self.x_c
        # self.y_test_t = self.y_t
        # self.y_test_c = self.y_c

        print("Preprocessed data has been fake split. It is attached as an attribute: .x_train_t, .x_train_c, .y_train_t, .y_train_c, .x_test_t, .x_test_c, .y_test_t, .y_test_c. They are all identical to x_t, x_c, y_t and y_c")

    def get_id_label(self):
        label_id = 'admissionid'
        return label_id

def label_data(data):
    
    # load data
    X = data.drop(columns=['admissionid', '28daymortality', 'high_dose_steroids']).columns.tolist()
    Y = '28daymortality'
    T = 'high_dose_steroids'

    return X, Y, T