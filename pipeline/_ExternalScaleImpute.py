import pickle

class Mixin:
    def external_scale_impute(self):
        # load scaling model
        with open(self.model_scaler_source, 'rb') as f:
            model_scaler = pickle.load(f)
        f.close()
        self.model_scaler = model_scaler

        # load imputer object
        with open(self.model_imputer_source, 'rb') as f:
            model_imputer = pickle.load(f)
        f.close
        self.model_imputer = model_imputer
        
        data = self.preprocessed_data

        # assign scaled df and scale it
        self.preprocessed_data_scaled = self.preprocessed_data.copy()
        self.preprocessed_data_scaled[self.label_x] = model_scaler.transform(self.preprocessed_data[self.label_x])

        # assign imputed df and impute it
        self.preprocessed_data_scaled_imputed = self.preprocessed_data_scaled.copy()
        self.preprocessed_data_scaled_imputed[self.label_x] = self.preprocessed_data_scaled[self.label_x].fillna(model_imputer)
