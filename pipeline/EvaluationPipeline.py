import pickle

import _Extraction
import _PreProcessing
import _FakeSplit
import _ExternalScaleImpute
import _ExternalFeatureSelection
import _LoadModel
import _Evaluate
import _Plotting
import _EffectEstimation
import _Interpretation

class EvaluationPipeLine(_Extraction.Mixin, _PreProcessing.Mixin, _FakeSplit.Mixin, _ExternalScaleImpute.Mixin, _ExternalFeatureSelection.Mixin, _LoadModel.Mixin, _Evaluate.Mixin, _Plotting.Mixin, _EffectEstimation.Mixin, _Interpretation.Mixin):
    def __init__(
            self, 
            name,
            model_scaler_source,
            model_imputer_source,
            selected_features_source=None,
            model_source=None,
            m0_source=None,
            m1_source=None,
            mx0_source=None,
            mx1_source=None,
            g_source=None,
            source="amsterdamumcdb", 
            algorithm="tarnet-imacfr", 
            random_state = 42):
        
        self.name = name
        self.source = source
        self.algorithm = algorithm
        self.random_state = random_state
        self.selected_features_source = selected_features_source
        
        self.preprocessed_data = None
        self.normalizing_model = None
        self.scaled_columns = None

        self.x_t = None
        self.x_c = None
        self.y_t = None
        self.y_c = None

        self.x_train_t = None
        self.x_train_c = None
        self.y_train_t = None
        self.y_train_c = None

        self.x_test_t = None
        self.x_test_c = None
        self.y_test_t = None
        self.y_test_c = None

        self.label_x = None
        self.label_y = None
        self.label_t = None

        self.tau_est = None

        # preprocess specifics
        self.model_scaler_source = model_scaler_source
        self.model_imputer_source = model_imputer_source
    
        # model sources
        self.model_source = model_source
        self.m0_source = m0_source
        self.m1_source = m1_source

        # TARNet attributes
        self.model = None
        self.head_loss = None
        self.m = None

        # T-learner attributes
        self.m0 = None
        self.m1 = None

        # X-learner attributes (and load g)
        if self.algorithm == 'x-learner':
            self.mx0_source = mx0_source
            with open(self.mx0_source, 'rb') as f:
                    self.mx0 = pickle.load(f)
            f.close()

            self.mx1_source = mx1_source
            with open(self.mx1_source, 'rb') as f:
                    self.mx1 = pickle.load(f)
            f.close()

            self.g_source = g_source
            with open(self.g_source, 'rb') as f:
                    self.g = pickle.load(f)
            f.close()

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

        # compatibility attributes
        self.metalearner_classifier = None

    # def extract_data(self): -> _Extraction
        
    # def preprocess_data(self): -> _Preprocess
        
    # def fake_split(self, random_state=42): -> _FakeSplit
        
    # def external_scale_impute(self) -> _ExternalScaleImpute
        
    # def load_model(self): -> _LoadModel

    # def evaluate(self): -> _Evaluation
        
    # def create_plot(self): -> _Plotting
        
    # def ite_estimation(self): -> _EffectEstimation
        
    # def interpretation(self): -> _Interpretation