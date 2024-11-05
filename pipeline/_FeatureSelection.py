from utils import ROOT
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
import numpy as np
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

import logging
logger = logging.getLogger('run')

class Mixin:
    def feature_selection(self):
        random_state = self.random_state
        feature_selection_model = 'LogisticRegression'

        model_dict = {
        'LogisticRegression': LogisticRegression(penalty='l1', solver='saga', C=2, multi_class='multinomial', n_jobs=-1, random_state=self.random_state)
        , 'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=200, max_depth=3, min_samples_leaf=.06, n_jobs=-1, random_state=self.random_state)
        , 'RandomForestClassifier': RandomForestClassifier(n_estimators=20, max_depth=2, min_samples_leaf=.1, random_state=self.random_state, n_jobs=-1)
        , 'XGBoostClassifier': XGBClassifier(random_state=self.random_state)
	    }

        estimator_dict = {}
        importance_fatures_sorted_all = pd.DataFrame()
        for model_name, model in model_dict.items():
            
            if model_name == 'XGBoostClassifier':
                x_train = self.preprocessed_data_scaled[self.label_x]
            if model_name != 'XGBoostClassifier':
                x_train = self.preprocessed_data_scaled_imputed[self.label_x]
            y_train = self.preprocessed_data_scaled_imputed[self.label_y]

            print('='*10, model_name, '='*10)
            model.fit(x_train, y_train)
            print('AUC in training:', roc_auc_score(y_train, model.predict(x_train)))
            #print('Accuracy in valid:', accuracy_score(model.predict(X_valid), y_valid))
            importance_values = np.absolute(model.coef_) if model_name == 'LogisticRegression' else model.feature_importances_
            importance_fatures_sorted = pd.DataFrame(importance_values.reshape([-1, len(x_train.columns)]), columns=x_train.columns).mean(axis=0).sort_values(ascending=False).to_frame()
            importance_fatures_sorted.rename(columns={0: 'feature_importance'}, inplace=True)
            importance_fatures_sorted['ranking']= importance_fatures_sorted['feature_importance'].rank(ascending=False)
            importance_fatures_sorted['model'] = model_name
            print('Show top important features:')
            print(importance_fatures_sorted.drop('model', axis=1))#.head(20))
            importance_fatures_sorted_all = pd.concat([importance_fatures_sorted_all, importance_fatures_sorted], ignore_index=True)  #importance_fatures_sorted_all.append(importance_fatures_sorted)
            estimator_dict[model_name] = model

        plt.title('Feature importance ranked by number of features by model')
        sns.lineplot(data=importance_fatures_sorted_all, x='ranking', y='feature_importance', hue='model')
        plt.xlabel("Number of features selected")

        # select top 20 features from xgboost
        selected_features_by_model = importance_fatures_sorted.iloc[0:20].index

        ### Stage 2 ###

        if feature_selection_model == 'XGBoostClassifier':
            x_train = self.preprocessed_data_scaled[self.label_x]
        if feature_selection_model != 'XGBoostClassifier':
            x_train = self.preprocessed_data_scaled_imputed[self.label_x]

        # it takes much more time comparing to the feature selelction by model
        rfecv = RFECV(estimator=model_dict[feature_selection_model].set_params(max_iter=150, C=1), step=1, cv=StratifiedShuffleSplit(1, test_size=.2, random_state=random_state), scoring='roc_auc', n_jobs=-1)
        rfecv.fit(x_train[selected_features_by_model], y_train)
        plt.figure()
        plt.title('Feature importance ranked by number of features by model')
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        rfecv_cv_score = rfecv.cv_results_.get('mean_test_score')
        plt.plot(range(1, len(rfecv_cv_score) + 1), rfecv_cv_score)
        plt.plot(rfecv.n_features_, rfecv_cv_score[rfecv.n_features_-1], marker='o', label='Optimal number of feature')
        plt.legend(loc='best')
        #plt.show()

        rfecv_df = pd.DataFrame({'col': selected_features_by_model}) 
        rfecv_df['rank'] = np.nan
        for index, support in enumerate(rfecv.get_support(indices=True)):
            rfecv_df.loc[support, 'rank'] = index
        for index, rank in enumerate(rfecv.ranking_ -2):
            if rank >= 0:
                rfecv_df.loc[index, 'rank'] = rfecv.n_features_ + rank
        rfecv_df = rfecv_df.sort_values(by=['rank'])
        print("")
        print("final features with ranking:")
        print(rfecv_df)
        rfecv_df.to_csv(ROOT + '\\saved_csvs\\features_ranked.csv')

        # get list of most important feature combinations -> 17 according to plot
        self.ranked_features = rfecv_df['col'].to_list()
              
        

        logger.info('.feature_selection ran succesfully')

