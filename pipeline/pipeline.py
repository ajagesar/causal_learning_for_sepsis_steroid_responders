import _Extraction
import _PreProcessing
import _Split
import _ScaleImpute
import _FeatureSelection
import _HyperparameterTuning
import _Modeling
import _Evaluate
import _Plotting
import _EffectEstimation
import _Interpretation

import logging
logger = logging.getLogger('run')

class PipeLine(_Extraction.Mixin, _PreProcessing.Mixin, _FeatureSelection.Mixin, _Split.Mixin, _ScaleImpute.Mixin, _HyperparameterTuning.Mixin, _Modeling.Mixin, _Evaluate.Mixin, _Plotting.Mixin, _EffectEstimation.Mixin, _Interpretation.Mixin):

    def __init__(self, name, source="amsterdamumcdb", algorithm="tarnet", metalearner_classifier=None, random_state = 42):
        
        logger.info('-------') # mark beginning of new log process

        self.name = name
        self.source = source
        self.algorithm = algorithm
        self.metalearner_classifier = metalearner_classifier
        self.random_state = random_state
        
        # data sources
        self.preprocessed_data = None
        self.preprocessed_data_scaled = None
        self.preprocessed_data_scaled_imputed = None

        # splitted ids
        self.treated_ids = None
        self.control_ids = None

        self.train_ids = None
        self.test_ids = None

        # splitted labels
        self.label_x = None
        self.label_x_selected = None
        self.label_y = None
        self.label_t = None
        self.label_id = None

        self.scaler = None
        self.scaled_columns = None

        self.selected_features = None

        # self.x_t_unscaled = None
        # self.x_c_unscaled = None
        # self.x_t = None
        # self.x_c = None
        # self.y_t = None
        # self.y_c = None

        # self.x_train = None
        # self.x_train_t = None
        # self.x_train_c = None

        # self.y_train = None
        # self.y_train_t = None
        # self.y_train_c = None

        # self.x_test = None
        # self.x_test_t = None
        # self.x_test_c = None

        # self.y_test = None
        # self.y_test_t = None
        # self.y_test_c = None

        self.tau_est = None
    
        # TARNet attributes
        self.model = None
        self.head_loss = None
        self.m = None

        # T-learner attributes
        self.m0 = None
        self.m1 = None

        # X-learner attributes
        self.mx0 = None
        self.mx1 = None
        self.g = None

        # predictions
        self.y_pred_test_t = None
        self.y_pred_test_c = None

        # counterfactuals of train set & test set
        self.y_cf_of_t = None
        self.y_cf_of_c = None
        self.y_cf_of_test_t = None
        self.y_cf_of_test_c = None

        # ite
        self.ite_t = None
        self.ite_c = None
        self.ite = None

        # define responders
        self.responders = None
        self.non_responders = None
        self.harmers = None

        # aggregated statistics
        self.responders_stats = None
        self.non_responders_stats = None
        self.harmers_stats = None

        # metrics
        self.confusion_matrix = None
        self.loss1_test = None
        self.loss2_test = None
        self.auc = None
        self.brier_score_loss = None
        
        # plots
        self.roc_curve = None
        self.calibration_curve = None

    # def extract_data(self): -> _Extraction
        
    # def preprocess_data(self): -> _Preprocess
        
    # def feature_selection(): -> _FeatureSelection
        
    # def split(self, random_state=42): -> _Split
        
    # def feature_selection(self): -> _Feature Selection
        
    # def hyperparameter_tuning(self): -> _HyperparameterTuning
        
    # def hyperparameter_tuning(self): -> _HyperparameterTuning

    # def train_model(self) -> _Modeling

    # def evaluate(self): -> _Evaluation
        
    # def create_plot(self): -> _Plotting
        
    # def ite_estimation(self): -> _EffectEstimation
        
    # def interpretation(self): -> _Interpretation
    


# TODO make conversion between tensors and pandas df scalable
# TODO split total, train and test set properly in code
    