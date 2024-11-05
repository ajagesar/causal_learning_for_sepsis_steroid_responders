import pickle

class Mixin:
    def external_feature_selection(self):
        with open(self.selected_features_source, 'rb') as f:
            label_x_selected = pickle.load(f)
        f.close()
    
        self.label_x_selected = label_x_selected