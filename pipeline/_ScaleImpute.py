from sklearn.preprocessing import MinMaxScaler
from utils import ROOT
import pickle

import logging
logger = logging.getLogger('run')

class Mixin:
    def scale_impute(self):
        # scaling
        data_to_fit_scaler = self.preprocessed_data.loc[self.preprocessed_data[self.label_id].isin(self.train_ids)] 

        scaler = MinMaxScaler()
        scaler.fit(data_to_fit_scaler[self.label_x])

        # assign scaled df:
        self.preprocessed_data_scaled = self.preprocessed_data.copy()
        self.preprocessed_data_scaled[self.label_x] = scaler.transform(self.preprocessed_data[self.label_x]) 

        self.scaler = scaler
        
        # save scaler for external validation
        with open(ROOT + f'\\saved_models\\{self.source}_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        f.close()
        
        #imputation by median (simple method)
            #code for imputation

        # get medians of train data
        medians_train = self.preprocessed_data_scaled.loc[self.preprocessed_data_scaled[self.label_id].isin(self.train_ids)][self.label_x].median()

        # save imputations for external validation
        with open(ROOT + f'\\saved_models\\{self.source}_imputer.pkl', 'wb') as f:
                pickle.dump(medians_train, f)
        f.close()

        # assign scaled imputed df
        self.preprocessed_data_scaled_imputed = self.preprocessed_data_scaled.copy()
        self.preprocessed_data_scaled_imputed[self.label_x] = self.preprocessed_data_scaled[self.label_x].fillna(medians_train) #medians based on train data
        
        logger.info('.scale_impute() ran succesfully')

        # self.x_train = self.x_train.fillna(self.x_train.median())
        # self.x_train_t = self.x_train_t.fillna(self.x_train.median())
        # self.x_train_c = self.x_train_c.fillna(self.x_train.median())

        # self.x_test = self.x_test.fillna(self.x_train.median()) # <- NOTE: imputed test set with median of train set to prevent leakage
        # self.x_test_t = self.x_test_t.fillna(self.x_train.median())
        # self.x_test_c = self.x_test_c.fillna(self.x_train.median())

        # self.x_train[columns_to_scale] = scaler.transform(self.x_train[columns_to_scale])
        # self.x_train_t[columns_to_scale] = scaler.transform(self.x_train_t[columns_to_scale])
        # self.x_train_c[columns_to_scale] = scaler.transform(self.x_train_c[columns_to_scale])

        # self.x_test[columns_to_scale] = scaler.transform(self.x_test[columns_to_scale])
        # self.x_test_t[columns_to_scale] = scaler.transform(self.x_test_t[columns_to_scale])
        # self.x_test_c[columns_to_scale] = scaler.transform(self.x_test_c[columns_to_scale])

        # make a copy for scaling
        # self.x_t = self.x_t_unscaled.copy()
        # self.x_c = self.x_c_unscaled.copy()

        # self.x_t[columns_to_scale] = scaler.transform(self.x_t_unscaled[columns_to_scale])
        # self.x_c[columns_to_scale] = scaler.transform(self.x_c_unscaled[columns_to_scale])


        # self.x_t = self.x_t.fillna(self.x_train.median())
        # self.x_c = self.x_c.fillna(self.x_train.median())

        #self.scaler = scaler

# def fit_normalization_model(inclusions):
#     columns_to_scale = inclusions.drop(columns=['admissionid'])
#     scaler = MinMaxScaler()
#     scaler.fit(columns_to_scale)

#     return scaler

# def impute_median(inclusions):
#     inclusions = inclusions.fillna(inclusions.median())
#     return inclusions

# def normalize_data(inclusions, normalizing_model):
#     columns_to_scale = inclusions.drop(columns=['admissionid']).columns
#     inclusions[columns_to_scale] = normalizing_model.transform(inclusions[columns_to_scale])
#     return inclusions, columns_to_scale