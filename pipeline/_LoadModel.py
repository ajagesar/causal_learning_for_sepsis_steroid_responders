import pickle
from utils import ROOT

class Mixin:
    def load_model(self):
        if (self.algorithm == 'tarnet') or (self.algorithm == 'tarnet-imacfr') or (self.algorithm == 's-learner'):
            with open(self.model_source, 'rb') as f:
                model = pickle.load(f)
            f.close()

            self.model = model

        if (self.algorithm == 't-learner') or (self.algorithm == 'x-learner'):
            with open(self.m0_source, 'rb') as f:
                m0 = pickle.load(f)
            f.close()

            with open(self.m1_source, 'rb') as f:
                m1 = pickle.load(f)
            f.close()

            self.m0 = m0
            self.m1 = m1