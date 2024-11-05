import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import ROOT
from scipy.stats import bootstrap

import logging
logger = logging.getLogger('run')
class Mixin:
    def interpret(self):

        # use unscaled x_t and x_c

        if self.algorithm == 'tarnet':
            # tensor to df
            ite_t = pd.DataFrame(self.ite_t.detach().numpy(), columns=['ite'])
            ite_c = pd.DataFrame(self.ite_c.detach().numpy(), columns=['ite'])

            # reconstruct original preprocessed df, but with added ite
            y_t = self.y_t.reset_index().drop(columns=['index'])
            y_c = self.y_c.reset_index().drop(columns=['index'])

            interpretation_df_t = pd.concat([self.x_t_unscaled.reset_index(), y_t, ite_t], axis=1)
            interpretation_df_c = pd.concat([self.x_c_unscaled.reset_index(), y_c, ite_c], axis=1)

            interpretation_df_t['high_dose_steroids'] = 1
            interpretation_df_c['high_dose_steroids'] = 0

            # combine treatment and control groups in one dataframe
            interpretation_df = pd.concat([interpretation_df_t, interpretation_df_c], axis=0)

            interpretation_df = interpretation_df.rename(columns={'index':'admissionid'})

        if self.algorithm == "tarnet-imacfr":
            ite = pd.DataFrame(self.ite, columns=['ite'])

            interpretation_df = pd.concat([self.preprocessed_data, ite], axis=1)

        if (self.algorithm == 't-learner') or (self.algorithm == 'x-learner') or (self.algorithm == 's-learner'):
            ite = pd.DataFrame(self.ite, columns=['ite'])

            interpretation_df = pd.concat([self.preprocessed_data, ite], axis=1)

        # fill missing with median for bootstrapping
        interpretation_df = interpretation_df.fillna(interpretation_df.median())

        # inverse transform scaling for interpretation
        #interpretation_df[self.scaled_columns] = self.scaler.inverse_transform(interpretation_df[self.scaled_columns])

        self.responders = interpretation_df.loc[interpretation_df['ite'] <= -0.1] # 10% lower chance of mortality compared with control
        self.non_responders = interpretation_df.loc[(interpretation_df['ite'] > -0.1) & (interpretation_df['ite'] < 0.1)] 
        self.harmers = interpretation_df.loc[interpretation_df['ite'] >= 0.1] # 10% higher chance of mortality compared with control

        self.ate = np.mean(interpretation_df['ite'])

        # create graph that Louk showed
        ites_np = interpretation_df['ite'].to_numpy()
        self.visualise_ites(
            ites=ites_np,
            target="mortality probability",
            intervention="steroids",
            split="whole dataset"
        )

        # create aggregated statistics for responders and non-responders
        self.responders_stats = self.responders.describe()
        self.non_responders_stats = self.non_responders.describe()
        self.harmers_stats = self.harmers.describe()
        
        # get bootstrapped results
        if len(self.responders) > 2:
            self.responders_bootstrapped_stats = self.get_bootstrapped_stats(df=self.responders)
            self.responders_bootstrapped_stats.to_markdown(ROOT + f'\\paper\\output\\{self.name}_{self.source}_responders_bootstrapped_stats.txt')

        if len(self.non_responders) > 2:
            self.non_responders_bootstrapped_stats = self.get_bootstrapped_stats(df=self.non_responders)
            self.non_responders_bootstrapped_stats.to_markdown(ROOT + f'\\paper\\output\\{self.name}_{self.source}_non_responders_bootstrapped_stats.txt')

        if len(self.harmers) > 2:
            self.harmers_bootstrapped_stats = self.get_bootstrapped_stats(df=self.harmers)
            self.harmers_bootstrapped_stats.to_markdown(ROOT + f'\\paper\\output\\{self.name}_{self.source}_harmers_bootstrapped_stats.txt')

        # sent to markdown txt file for paper
        self.responders_stats.to_markdown(ROOT + f'\\paper\\output\\{self.name}_{self.source}_responders_stats.txt')
        self.non_responders_stats.to_markdown(ROOT + f'\\paper\\output\\{self.name}_{self.source}_non_responders_stats.txt')
        self.harmers_stats.to_markdown(ROOT + f'\\paper\\output\\{self.name}_{self.source}_harmers_stars.txt')     

        logger.info('.interpret() ran succesfully')

    def visualise_ites(self, ites, target, intervention, split="test"):

        ites = [i[0] for i in ites.reshape(-1,1)]
        ites = sorted(list(ites))

        x_values = range(1, len(ites) + 1)
        stddev = np.std(ites)
        upper_bound = [val + stddev for val in ites]
        lower_bound = [val - stddev for val in ites]

        sns.set_theme(style="whitegrid")

        #plt.figure(figsize=(8,4))
        plt.figure(figsize=(12,6))
        plt.fill_between(x_values, lower_bound, upper_bound, color='gray', alpha=0.2, label="standard deviation")
        plt.axhline(y=np.mean(ites), color='red', linestyle='--', label="ATE estimate")
        plt.title(f"Orded ITE-estimates for {intervention}-{target} pair on {split}.")
        sns.lineplot(x=x_values, y=ites, color='b', label='predicted ITE values', lw=2.5)

        # experimental code to add lines to values above and below 0.1 (responders and none responders)

        above_threshold = [item for item in ites if item > 0.1]
        below_threshold = [item for item in ites if item < -0.1]

        #check if list is empy before drawing extra lines to mark responders and non responders. Also make it a sorted list
        if above_threshold:
            above_threshold = sorted(list(above_threshold))

            x_start = len(ites) - len(above_threshold)
            x_end = len(ites)

            x_values = range(x_start, x_end)
            sns.lineplot(x=x_values, y=above_threshold, color='orange', label='steroid harmers', lw=2.5)

        if below_threshold: 
            below_threshold = sorted(list(below_threshold))
            x_values = range(1, len(below_threshold) + 1) # TODO to fix
            sns.lineplot(x=x_values, y=below_threshold, color='green', label='steroid responders', lw=2.5)

        plt.savefig(ROOT + f'\\saved_csvs\\{self.name}_ite_curve.png')
        plt.show()

    def get_bootstrapped_stats(self, df, n_resamples=1000):
        
        features = df.columns

        feature_list = []
        feature_mean_list = []
        feature_median_list = []
        bootstrap_cis_low = []
        bootstrap_cis_high = []

        # preprare dataset
        for feature in df.columns:
            sample = (df[feature].to_numpy(),)#.flatten()

            # get bootstrapped data set
            bootstrap_result = bootstrap(sample, statistic=np.mean, n_resamples=n_resamples, confidence_level=0.95, method='percentile', random_state=self.random_state)

            bootstrap_ci_low = bootstrap_result.confidence_interval.low
            bootstrap_ci_high = bootstrap_result.confidence_interval.high

            bootstrap_cis_low.append(bootstrap_ci_low)
            bootstrap_cis_high.append(bootstrap_ci_high)

            feature_list.append(feature)
            feature_mean_list.append(df[feature].mean())
            feature_median_list.append(df[feature].median())
        
        df_cis = pd.DataFrame({
            'feature':feature_list,
            'mean': feature_mean_list,
            'median': feature_median_list,
            'low': bootstrap_cis_low,
            'high': bootstrap_cis_high
        })
            
        # variables = df.columns

        # list_of_means = []

        # for i in range(loops):
        #     df_bootstrapped = df.sample(frac=1, replace=True, random_state=self.random_state)

        #     df_bootstrapped_mean = df_bootstrapped.mean().to_list()

        #     list_of_means.append(df_bootstrapped_mean)
        
        # df_bootstrapped_mean = pd.DataFrame(list_of_means, columns=variables)

        # get CI


        # get calculations
        #df_bootstrapped

        #df_bootstrapped_stats = None

        return df_cis


        
