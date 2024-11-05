from utils import ROOT
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import logging
logger = logging.getLogger('run')

class Mixin:
    def preprocess_data(self):
        if self.source == "amsterdamumcdb":
            steroids_high_dose_ids = preprocess_steroids()
            inclusions = preprocess_inclusions()
            inclusions = define_intervention_group(steroids_high_dose_ids=steroids_high_dose_ids, inclusions=inclusions)
            covariates = preprocess_covariates()
            inclusions = add_covariates(inclusions=inclusions, covariates=covariates)
            preprocessed_data = inclusions.copy()

        if self.source == "mimic":
            mortality, sepsis, demographics, blood_gas, features, max_gamma, ventilation, crp, steroids = load_mimic()
            crp = add_stay_id(mortality, crp)
            mortality = define_28d_mortality(mortality)
            steroids = clean_steroids(steroids)
            df_mimic = merge_df_mimic(mortality, sepsis, demographics, blood_gas, features, max_gamma, ventilation, crp, steroids)
            df_mimic = clean_df_mimic(df_mimic)
            df_mimic = convert_to_aumcdb_units(df_mimic)  
            df_mimic.to_csv(ROOT + '\\saved_csvs\\mimic_unimputed.csv')
            df_mimic = df_mimic_assumptions(df_mimic)
            df_mimic = rename_df_mimic(df_mimic)
            preprocessed_data = df_mimic.copy()

        # # imputation and normalizing -> separate script for only working with training data
        # normalizing_model = fit_normalization_model(preprocessed_data)
        # preprocessed_data = impute_median(inclusions=preprocessed_data) # TODO impute with normal value ?
        # preprocessed_data, columns_to_scale = normalize_data(inclusions=preprocessed_data, normalizing_model=normalizing_model)
        
        preprocessed_data.to_csv(ROOT + f'\\saved_csvs\\{self.source}_preprocessed_data.csv', index=False)

        # self.scaled_columns = columns_to_scale
        # self.normalizing_model = normalizing_model

        self.preprocessed_data = preprocessed_data

        print("data was written to csv and attached to attribute .preprocessed_data")

        logger.info('preprocess_data() ran succesfully')

def preprocess_steroids(): # TODO steroids high dose & all or nothing
    # load data
    steroids = pd.read_csv(ROOT + "\\saved_csvs\\steroids.csv", index_col=False)
    
    # make distinction between high dose steroids and no/low dose steroids
    # 200mg hydrocort = 50mg prednison = 8mg dexa = 40mg methylprednison
    steroids['starttime'] = pd.to_datetime(steroids['starttime'], unit='ms')
    steroids['stoptime'] = pd.to_datetime(steroids['stoptime'], unit='ms')

    # resample data to bins of 1 day and flatten for processing
    steroids_resampled = steroids.groupby(['admissionid', 'itemid']).resample(rule='D', on='starttime')['administered'].sum().reset_index()

    """
    define high dose steroids ids where:
    - hydrocort (itemid = 7106) > 200mg / day
    - prednison (itemid = 6922) > 50mg / day
    - methylprednisolon (itemid = 8132) > 40mg / day
    - dexamethason (itemid = 6995) > 8mg / day
    """

    steroids_resampled['high_dose_steroids'] = 0

    steroids_resampled.loc[(steroids_resampled['itemid'] == 7106) & (steroids_resampled['administered'] >= 200), 'high_dose_steroids'] = 1
    steroids_resampled.loc[(steroids_resampled['itemid'] == 6922) & (steroids_resampled['administered'] >= 50), 'high_dose_steroids'] = 1
    steroids_resampled.loc[(steroids_resampled['itemid'] == 8132) & (steroids_resampled['administered'] >= 40), 'high_dose_steroids'] = 1
    steroids_resampled.loc[(steroids_resampled['itemid'] == 6995) & (steroids_resampled['administered'] >= 8), 'high_dose_steroids'] = 1

    steroids_high_dose_ids = steroids_resampled.loc[steroids_resampled['high_dose_steroids'] == 1]['admissionid'].drop_duplicates() # <- all admissions that received high dose steroids according to specs of above

    return steroids_high_dose_ids


def preprocess_inclusions():
    # load inclusions: sepsis patients
    inclusions = pd.read_csv(ROOT + "\\saved_csvs\\sepsis.csv", index_col=False)

    # extract only relevant columns from sepsis patients
    inclusions = inclusions[['admissionid', 'gender', 'agegroup', 'admittedat', 'dischargedat', 'dateofdeath', 'weightgroup', 'heightgroup']].copy()

    # remove patients with missing genders
    inclusions = inclusions.dropna(subset='gender')
    inclusions = inclusions.loc[(inclusions['gender'] == 'Man') | (inclusions['gender'] == 'Vrouw')]
    inclusions.loc[inclusions['gender'] == 'Man', 'gender'] = 1
    inclusions.loc[inclusions['gender'] == 'Vrouw', 'gender'] = 0

    # calculate 28 day mortality -> 28 days = 2 419 200 000 ms
    inclusions['28daymortality'] = 0
    inclusions.loc[(inclusions['dateofdeath'] - inclusions['admittedat']) < 2419200000, '28daymortality'] = 1

    # now only keep relevant columns from inclusions
    inclusions = inclusions[['admissionid', 'gender', 'agegroup', 'weightgroup', 'heightgroup', '28daymortality']].copy().set_index('admissionid')

    # convert age, weight and height; fill missing with median
    age_dict = {
        '18-39': 29,
        '40-49': 45,
        '50-59': 55,
        '60-69': 65,
        '70-79': 75,
        '80+': 85
    }

    weight_dict = {
        '59-': 55,
        '60-69': 65,
        '70-79': 75,
        '80-89': 85,
        '90-99': 95,
        '100-109': 105,
        '110+': 115
    }

    height_dict = {
        '159-': 155,
        '160-169': 165,
        '170-179': 175,
        '180-189': 185,
        '190+': 195
    }

    inclusions = inclusions.replace({"agegroup": age_dict})
    inclusions = inclusions.replace({"weightgroup": weight_dict})
    inclusions = inclusions.replace({"heightgroup": height_dict})

    # save unimputed data for supplementary materials
    inclusions_demographics_unimputed = inclusions.copy()
    inclusions_demographics_unimputed.to_csv(ROOT + '\\saved_csvs\\demographics_unimputed.csv')

    inclusions = inclusions.rename(columns={"agegroup":"estimated_age", "weightgroup":"estimated_weight", "heightgroup":"estimated_height"})
    inclusions[['estimated_age', 'estimated_weight', 'estimated_height']] = inclusions[['estimated_age', 'estimated_weight', 'estimated_height']].fillna(inclusions[['estimated_age', 'estimated_weight', 'estimated_height']].median())

    inclusions = inclusions.reset_index()

    return inclusions


def define_intervention_group(inclusions, steroids_high_dose_ids):
    # assign values to steroids given to patients
    inclusions['high_dose_steroids'] = 0
    inclusions.loc[inclusions['admissionid'].isin(steroids_high_dose_ids), 'high_dose_steroids'] = 1

    # remove duplicates (e.g. patients who first received methylprednisolon and later regular prednisolon)
    inclusions = inclusions.drop_duplicates(subset='admissionid')

    return inclusions

def preprocess_covariates():
    features = pd.read_csv(ROOT + "\\saved_csvs\\features.csv", index_col=False)
    mechanical_ventilation = pd.read_csv(ROOT + "\\saved_csvs\\mechanical_ventilation.csv", index_col=False)
    vaso = pd.read_csv(ROOT + "\\saved_csvs\\vaso.csv", index_col=False)
    pf = pd.read_csv(ROOT + "\\saved_csvs\\pf.csv", index_col=False)
    ph = pd.read_csv(ROOT + "\\saved_csvs\\ph.csv", index_col=False)


    # VASO
    # aggregate to get max gamma per admission
    vaso = vaso.groupby(['admissionid']).agg(
    total_duration=pd.NamedAgg(column='duration', aggfunc='sum'),
    max_gamma=pd.NamedAgg(column='gamma', aggfunc='max')
    ).reset_index()

    # general features
    # rename various itemid as one
    renaming_dict = {
        'sodium':[6840, 9555, 9924, 10284],
        'potassium':[6835, 9556, 9927, 10285],
        'creatinin':[6836, 9941, 14216],
        'leucocytes':[6779, 9965],
        'trombocytes':[9964, 6797, 10409, 14252],
        'temperature':[8658, 8659, 8662, 13058, 13059, 13060, 13061, 13062, 13063, 13952, 16110],
        'glucose':[9557, 6833, 9947],
        'bun':[6850, 9943],
        'crp':[6825, 10079],
        'bicarbonate':[6810, 9992, 9993],
        'heartfrequency':[6640],
        'lactate':[10053, 6837, 9580]
    }

    # loop over dictionary and rename features accordingly
    features['feature'] = None

    for i in renaming_dict.items():
        features.loc[features['itemid'].isin(i[1]), 'feature'] = i[0]

    # cleanup: based on amsterdamumcdb severity scores notebooks
    # sodium: remove extreme outliers, most likely data entry errors (manual_entry = True)
    features.loc[(features['feature']=='sodium') & (features['value'] < 95) & (features['validated']==True), 'value'] = np.NaN
    features.loc[(features['feature']=='sodium') & (features['value'] > 165) & (features['validated']==True), 'value'] = np.NaN

    # potassium: remove extreme outliers, most likely data entry errors (manual_entry = True)
    features.loc[(features['feature']=='potassium') & (features['value'] > 8) & (features['validated'] == True), 'value'] = np.NaN
    features.loc[(features['feature']=='potassium') & (features['value'] < 2) & (features['validated'] == True), 'value'] = np.NaN
    features.loc[(features['feature']=='potassium') & (features['value'] <= 2.2) & (features['validated'] == False), 'value'] = np.NaN
    features.loc[(features['feature']=='potassium') & (features['value'] > 10) & (features['validated'] == False), 'value'] = np.NaN

    # creatinine: remove extreme outliers, most likely data entry errors (manual_entry = True)
    features.loc[(features['feature']=='creatinin') & (features['value'] < 30) & (features['validated'] == True), 'value'] = np.NaN

    # leucocytes: separator error: e.g. 117 > 11.7
    features.loc[(features['feature']=='leucocytes') & (features['value'] > 70) & (features['validated'] == True), 'value'] = \
        features.loc[(features['feature']=='leucocytes') & (features['value'] > 70) & (features['validated'] == True), 'value'] / 10

    # trombocytes: no cleanup

    # temperature:
    # 1. assumes temperatures  > 100 have been entered without a decimal separator
    features.loc[(features['feature']=='temperature') & (features['value'] <= 30), 'value'] = np.NaN
    # 2. remove extreme outliers, most likely data entry errors or measurement errors (i.e. ambient temperature due to displaced sensor)
    features.loc[(features['feature']=='temperature') & (features['value'] > 43), 'value'] = np.NaN

    # bun: aboven 70 seems very unrealistic based on expert opinion
    features.loc[(features['feature']=='bun') & (features['value'] > 70), 'value'] = np.NaN

    # crp: no cleanup

    # bicarbonate: based on plots in separate notebook
    features.loc[(features['feature']=='bicarbonate') & (features['value'] < 5), 'value'] =np.NaN
    features.loc[(features['feature']=='bicarbonate') & (features['value'] > 50), 'value'] =np.NaN

    # drop heart rate > 300 as aumcdb repository
    features.loc[(features['feature']=='heartfrequency') & (features['value'] > 300), 'value'] = np.NaN

    # lactate: no cleanup

    features = features.dropna(subset='value')

    # aggregate
    features = features.groupby(['admissionid', 'feature']).agg({'value':['median', 'max', 'min']}).reset_index()

    # drop multi-index from columns to be able to pivot and rename columns accordingly
    features = features.droplevel(0, axis=1).set_axis(['admissionid', 'feature', 'median', 'max', 'min'], axis=1)

    # pivot to proper format
    features = features.pivot(index='admissionid', columns='feature', values=['median', 'max', 'min'])

    # flatten pivoted table for easier wrangling with code below
    # code from https://stackoverflow.com/questions/27071661/pivoting-pandas-dataframe-into-prefixed-cols-not-a-multiindex
    features.columns = [' '.join(col).strip() for col in features.columns.values]

    # select columns to keep
    features = features[[ # removes min leucocytes and min sodium
        'median heartfrequency',
        'max bun',
        'max creatinin',
        'max crp',
        'max glucose',
        'max lactate',
        'max leucocytes',
        'max potassium',
        'max sodium',
        'max temperature',
        'min bicarbonate'
    ]]

    # PF-ratio:
    # remove extreme outliers
    pf.loc[(pf['fio2'] > 100), 'fio2'] = np.NaN

    # convert FiO2 in % to fraction
    pf.loc[(pf['fio2'] <= 100) & (pf['fio2'] >= 20) , 'fio2'] = pf['fio2']/100

    # remove extreme outliers (FiO2) (possible O2 flow?)
    pf.loc[(pf['fio2'] > 1), 'fio2'] = np.NaN

    # remove lower outliers, most likely incorrectly labeled as 'arterial' instead of '(mixed/central) venous'
    pf.loc[pf['pao2'] < 50, 'pao2'] = np.NaN
    pf = pf.dropna(subset=['pao2'])

    pf.loc[:,'pf_ratio'] = pf['pao2']/pf['fio2']

    # keep worst PF value
    pf = pf.groupby(by=['admissionid'])['pf_ratio'].min().reset_index()

    # pH
    # remove extreme outliers, most likely data entry errors (manual_entry = True) (notebooks aumcdb repository)
    ph.loc[(ph['value'] < 6.8) & (ph['manual_entry'] == True), 'value'] = np.NaN
    ph.loc[(ph['value'] > 7.6) & (ph['manual_entry'] == True), 'value'] = np.NaN
    ph = ph.dropna(subset=['value'])

    # keep worst arterial pH
    ph = ph.groupby(by=['admissionid'])['value'].min().reset_index()
    # rename 'value' to arterial_ph for merging
    ph = ph.rename(columns={"value":"arterial_ph"})

    # merging all into table called covariates
    covariates = pd.read_csv(ROOT + "\\saved_csvs\\covariate_framework.csv", index_col=False)

    # merge mechanical ventilation
    covariates = pd.merge(left=covariates, right=mechanical_ventilation[['admissionid', 'mechanical_ventilation_bool']], how='left', on='admissionid')
    ## send to separate csv for imputation calculation
    covariates['mechanical_ventilation_bool'].to_csv(ROOT + '\\saved_csvs\\ventilation_unimputed.csv')

    # label admissions with no registered mechanical ventilation support as False
    covariates.loc[covariates['mechanical_ventilation_bool'].isna(), 'mechanical_ventilation_bool'] = False
    
    # merge pf
    covariates = pd.merge(left=covariates, right=pf[['admissionid', 'pf_ratio']], how='left', on='admissionid')
    ## send to separate csv for imputation calculation
    covariates['pf_ratio'].to_csv(ROOT + '\\saved_csvs\\pf_ratio_unimputed.csv')

    # label missing pf ratios as normal (400)
    covariates = covariates.fillna(value={"pf_ratio":400})

    # merge vasopressor gamma
    covariates = pd.merge(left=covariates, right=vaso[['admissionid', 'max_gamma']], how='left', on='admissionid')
    ## send to separate csv for imputation calculation
    covariates['max_gamma'].to_csv(ROOT + '\\saved_csvs\\vaso_unimputed.csv')

    # impute gamma values with 0 if there is none being given
    covariates = covariates.fillna(value={"max_gamma":0})

    # merge with arterial phh
    covariates = pd.merge(left=covariates, right=ph[['admissionid', 'arterial_ph']], how='left', on='admissionid')

    # merge with other covariates
    covariates = pd.merge(left=covariates, right=features, how='left', on='admissionid')

    return covariates

def add_covariates(inclusions, covariates):
    # merge with existing table
    inclusions = pd.merge(left=inclusions, right=covariates, how='left', on='admissionid')
    inclusions = inclusions.drop(columns=['Unnamed: 0'])

    return inclusions

def remove_outliers(inclusions): # TODO add
    pass

def normal_value_imputation(inclusions):
    pass
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

def load_mimic():
    # load data
    mortality = pd.read_csv(ROOT + '\\saved_csvs\\mimic_mortality.csv', index_col=0)
    sepsis = pd.read_csv(ROOT + '\\saved_csvs\\mimic_sepsis.csv', index_col=0)
    demographics = pd.read_csv(ROOT + "\\saved_csvs\\mimic_demographics.csv", index_col=0)
    blood_gas = pd.read_csv(ROOT + '\\saved_csvs\\mimic_blood_gas_2.csv', index_col=0) # TODO change this to the other blood gas
    features = pd.read_csv(ROOT + '\\saved_csvs\\mimic_features.csv', index_col=0)
    max_gamma = pd.read_csv(ROOT + '\\saved_csvs\\mimic_max_gamma.csv', index_col=0)
    ventilation = pd.read_csv(ROOT + '\\saved_csvs\\mimic_ventilation.csv', index_col=0)

    # without stay_id
    crp = pd.read_csv(ROOT + "\\saved_csvs\\mimic_crp.csv", index_col=0)
    steroids = pd.read_csv(ROOT + "\\saved_csvs\\mimic_steroids.csv", index_col=0)

    return mortality, sepsis, demographics, blood_gas, features, max_gamma, ventilation, crp, steroids
    
def add_stay_id(mortality, crp):
    whole_data = pd.merge(left=mortality, right=crp, how='outer', on=['subject_id', 'hadm_id'])

    whole_data['charttime'] = pd.to_datetime(whole_data['charttime'])
    whole_data['icu_intime'] = pd.to_datetime(whole_data['icu_intime'])

    # only keep records with a charttime within 24h of icu.intime
    whole_data = whole_data.loc[(whole_data['charttime'] - whole_data['icu_intime'] <= pd.Timedelta(24, unit='h')) & (whole_data['charttime'] >= whole_data['icu_intime'])]
    
    # keep relevant crp data
    whole_data = whole_data[['stay_id', 'crp']]
    whole_data = whole_data.groupby('stay_id').agg({'crp':'max'})
    return whole_data

def define_28d_mortality(mortality):
    
    mortality['dod'] = pd.to_datetime(mortality['dod'])
    mortality['icu_intime'] = pd.to_datetime(mortality['icu_intime'])

    mortality['28daymortality'] = 0
    mortality.loc[mortality['dod'] - mortality['icu_intime'] <= pd.Timedelta(28, unit='day'), '28daymortality'] = 1

    return mortality

def clean_steroids(steroids):
    
    steroids['high_dose_steroids'] = 0
    steroids.loc[steroids['methylprednisolone_equivalent_total'] >= 120, 'high_dose_steroids'] = 1

    return steroids

### BELOW MIMIC-IV ###

def merge_df_mimic(mortality, sepsis, demographics, blood_gas, features, max_gamma, ventilation, crp, steroids):
    
    sepsis = sepsis.drop(columns=['subject_id'])
    demographics = demographics.drop(columns=['gender'])

    df_mimic = pd.merge(left=sepsis, right=mortality, on='stay_id', how='left')
    df_mimic = pd.merge(left=df_mimic, right=demographics, on='stay_id', how='left')
    df_mimic = pd.merge(left=df_mimic, right=blood_gas, on='stay_id', how='left')
    df_mimic = pd.merge(left=df_mimic, right=features, on='stay_id', how='left')
    df_mimic = pd.merge(left=df_mimic, right=max_gamma, on='stay_id', how='left')
    df_mimic = pd.merge(left=df_mimic, right=ventilation, on='stay_id', how='left')
    df_mimic = pd.merge(left=df_mimic, right=crp, on='stay_id', how='left')
    df_mimic = pd.merge(left=df_mimic, right=steroids, on='stay_id', how='left')

    return df_mimic

def clean_df_mimic(df_mimic):

    columns_to_keep = [
        'stay_id',
        'gender',
        'age',
        'weight',
        'height',
        '28daymortality',
        'high_dose_steroids',
        'ventilation',
        'min_pf_ratio',
        'max_gamma_vasopressor',
        'min_ph',
        'mean_heart_rate',
        'max_bun',
        'max_creatinine',
        'crp',
        'max_glucose',
        'max_lactate',
        'max_wbc',
        'max_potassium',
        'max_sodium',
        'max_temperature',
        'min_bicarbonate'
    ]

    # DONE: columns to get: 28d mortality, high dose steroids, max crp

    df_mimic = df_mimic[columns_to_keep]
    
    gender_dict = {
        'F':0,
        'M':1
    }

    df_mimic = df_mimic.replace({'gender': gender_dict})

    return df_mimic

def convert_to_aumcdb_units(df_mimic):
    # BUN from mg/dL to mmol/L
    df_mimic['max_bun'] = df_mimic['max_bun'] * 0.357

    # creatinin from mg/dL to micromol/L
    df_mimic['max_creatinine'] = df_mimic['max_creatinine'] * 88.4

    # glucose from mg/dL to mmol/L
    df_mimic['max_glucose'] = df_mimic['max_glucose'] * 0.0555

    return df_mimic

def df_mimic_assumptions(df_mimic):
    # impute missing values with following assumptions:
    df_mimic.loc[df_mimic['high_dose_steroids'].isna(), 'high_dose_steroids'] = 0 # no steroids given
    df_mimic.loc[df_mimic['min_pf_ratio'].isna(), 'min_pf_ratio'] = 400 # normal pf ratio of 400
    df_mimic.loc[df_mimic['max_gamma_vasopressor'].isna(), 'max_gamma_vasopressor'] = 0
    
    # TODO removing outliers

    return df_mimic

def rename_df_mimic(df_mimic):
    # change all names to the same names in amsterdamumcdb to keep the code the same

    rename_dict = {
        'stay_id':'admissionid',
        'gender':'gender',
        'age':'estimated_age',
        'weight':'estimated_weight',
        'height':'estimated_height',
        '28daymortality':'28daymortality',
        'high_dose_steroids':'high_dose_steroids',
        'ventilation':'mechanical_ventilation_bool',
        'min_pf_ratio':'pf_ratio',
        'max_gamma_vasopressor':'max_gamma',
        'min_ph':'arterial_ph',
        'mean_heart_rate':'median heartfrequency',
        'max_bun':'max bun',
        'max_creatinine':'max creatinin',
        'crp':'max crp',
        'max_glucose':'max glucose',
        'max_lactate':'max lactate',
        'max_wbc':'max leucocytes',
        'max_potassium':'max potassium',
        'max_sodium':'max sodium',
        'max_temperature':'max temperature',
        'min_bicarbonate':'min bicarbonate'
    }

    df_mimic = df_mimic.rename(columns=rename_dict)

    return df_mimic



# TODO unit conversion
