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

