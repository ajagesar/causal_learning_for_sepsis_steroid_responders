# set working directory
from utils import ROOT, device, get_overall_plot, get_summary_results
import os
os.chdir(ROOT)
import logging

import PipeLine as pl
import EvaluationPipeline as evalpl

logger = logging.getLogger('run')
logger.setLevel(logging.INFO)

formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s:%(name)s:%(message)s:', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler(ROOT + f'\\logs\\run.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# TARnetIMA CFRClassifier pipeline
imaf_pipeline = pl.PipeLine(name="tarnet-imacfr", source="amsterdamumcdb", algorithm="tarnet-imacfr", random_state=0)
imaf_pipeline.extract_data()
imaf_pipeline.preprocess_data() 
imaf_pipeline.split()
imaf_pipeline.scale_impute()
imaf_pipeline.feature_selection()
imaf_pipeline.hyperparameter_tuning()
imaf_pipeline.train_model()
imaf_pipeline.evaluate()
imaf_pipeline.create_plot()
imaf_pipeline.ite_estimation()
imaf_pipeline.interpret()

# 1. s_learner_lr_aumcdb
s_learner_lr_aumcdb = pl.PipeLine(
    name='s-learner_aumcdb', 
    source='amsterdamumcdb', 
    algorithm='s-learner', 
    metalearner_classifier='LR', 
    random_state=0
    )
s_learner_lr_aumcdb.extract_data()
s_learner_lr_aumcdb.preprocess_data()
s_learner_lr_aumcdb.split()
s_learner_lr_aumcdb.scale_impute()
s_learner_lr_aumcdb.feature_selection()
s_learner_lr_aumcdb.train_model()
s_learner_lr_aumcdb.evaluate()
s_learner_lr_aumcdb.create_plot()
s_learner_lr_aumcdb.ite_estimation()
s_learner_lr_aumcdb.interpret()

# 2. s_learner_xgb_aumcdb
s_learner_xgb_aumcdb = pl.PipeLine(
    name='s-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='s-learner',
    metalearner_classifier='XGB',
    random_state=0
)
s_learner_xgb_aumcdb.extract_data()
s_learner_xgb_aumcdb.preprocess_data()
s_learner_xgb_aumcdb.split()
s_learner_xgb_aumcdb.scale_impute()
s_learner_xgb_aumcdb.feature_selection()
s_learner_xgb_aumcdb.train_model()
s_learner_xgb_aumcdb.evaluate()
s_learner_xgb_aumcdb.create_plot()
s_learner_xgb_aumcdb.ite_estimation()
s_learner_xgb_aumcdb.interpret()

# 3. s_learner_mlp_aumcdb
s_learner_mlp_aumcdb = pl.PipeLine(
    name='s-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='s-learner',
    metalearner_classifier='MLP',
    random_state=0
)
s_learner_mlp_aumcdb.extract_data()
s_learner_mlp_aumcdb.preprocess_data()
s_learner_mlp_aumcdb.split()
s_learner_mlp_aumcdb.scale_impute()
s_learner_mlp_aumcdb.feature_selection()
s_learner_mlp_aumcdb.train_model()
s_learner_mlp_aumcdb.evaluate()
s_learner_mlp_aumcdb.create_plot()
s_learner_mlp_aumcdb.ite_estimation()
s_learner_mlp_aumcdb.interpret()

# 4. t_learner_lr_aumcdb
t_learner_lr_aumcdb = pl.PipeLine(
    name='t-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='t-learner',
    metalearner_classifier='LR',
    random_state=0
)
t_learner_lr_aumcdb.extract_data()
t_learner_lr_aumcdb.preprocess_data()
t_learner_lr_aumcdb.split()
t_learner_lr_aumcdb.scale_impute()
t_learner_lr_aumcdb.feature_selection()
t_learner_lr_aumcdb.train_model()
t_learner_lr_aumcdb.evaluate()
t_learner_lr_aumcdb.create_plot()
t_learner_lr_aumcdb.ite_estimation()
t_learner_lr_aumcdb.interpret()

# 5. t_learner_xgb_aumcdb
t_learner_xgb_aumcdb = pl.PipeLine(
    name='t-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='t-learner',
    metalearner_classifier='XGB',
    random_state=0
)
t_learner_xgb_aumcdb.extract_data()
t_learner_xgb_aumcdb.preprocess_data()
t_learner_xgb_aumcdb.split()
t_learner_xgb_aumcdb.scale_impute()
t_learner_xgb_aumcdb.feature_selection()
t_learner_xgb_aumcdb.train_model()
t_learner_xgb_aumcdb.evaluate()
t_learner_xgb_aumcdb.create_plot()
t_learner_xgb_aumcdb.ite_estimation()
t_learner_xgb_aumcdb.interpret()

# 6. t_learner_mlp_aumcdb
t_learner_mlp_aumcdb = pl.PipeLine(
    name='t-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='t-learner',
    metalearner_classifier='MLP',
    random_state=0
)
t_learner_mlp_aumcdb.extract_data()
t_learner_mlp_aumcdb.preprocess_data()
t_learner_mlp_aumcdb.split()
t_learner_mlp_aumcdb.scale_impute()
t_learner_mlp_aumcdb.feature_selection()
t_learner_mlp_aumcdb.train_model()
t_learner_mlp_aumcdb.evaluate()
t_learner_mlp_aumcdb.create_plot()
t_learner_mlp_aumcdb.ite_estimation()
t_learner_mlp_aumcdb.interpret()

# 7. x_learner_lr_aumcdb
x_learner_lr_aumcdb = pl.PipeLine(
    name='x-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='x-learner',
    metalearner_classifier='LR',
    random_state=0
)
t_learner_lr_aumcdb.extract_data()
x_learner_lr_aumcdb.preprocess_data()
x_learner_lr_aumcdb.split()
x_learner_lr_aumcdb.scale_impute()
x_learner_lr_aumcdb.feature_selection()
x_learner_lr_aumcdb.train_model()
x_learner_lr_aumcdb.evaluate()
x_learner_lr_aumcdb.create_plot()
x_learner_lr_aumcdb.ite_estimation()
x_learner_lr_aumcdb.interpret()

# 8. x_learner_xgb_aumcdb
x_learner_xgb_aumcdb = pl.PipeLine(
    name='x-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='x-learner',
    metalearner_classifier='XGB',
    random_state=0
)
x_learner_xgb_aumcdb.extract_data()
x_learner_xgb_aumcdb.preprocess_data()
x_learner_xgb_aumcdb.split()
x_learner_xgb_aumcdb.scale_impute()
x_learner_xgb_aumcdb.feature_selection()
x_learner_xgb_aumcdb.train_model()
x_learner_xgb_aumcdb.evaluate()
x_learner_xgb_aumcdb.create_plot()
x_learner_xgb_aumcdb.ite_estimation()
x_learner_xgb_aumcdb.interpret()

# 9. x_learner_mlp_aumcdb
x_learner_mlp_aumcdb = pl.PipeLine(
    name='x-learner_aumcdb',
    source='amsterdamumcdb',
    algorithm='x-learner',
    metalearner_classifier='MLP',
    random_state=0
)
x_learner_mlp_aumcdb.extract_data()
x_learner_mlp_aumcdb.preprocess_data()
x_learner_mlp_aumcdb.split()
x_learner_mlp_aumcdb.scale_impute()
x_learner_mlp_aumcdb.feature_selection()
x_learner_mlp_aumcdb.train_model()
x_learner_mlp_aumcdb.evaluate()
x_learner_mlp_aumcdb.create_plot()
x_learner_mlp_aumcdb.ite_estimation()
x_learner_mlp_aumcdb.interpret()

### EVALUATION ####

# 1. s_learner_lr_mimic
s_learner_lr_mimic = evalpl.EvaluationPipeLine(
    name='s-learner_mimic', 
    model_source=(ROOT + '\\saved_models\\amsterdamumcdb_s-learner_LR_model.pkl'),
    source='mimic',
    algorithm='s-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\s-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
s_learner_lr_mimic.extract_data()
s_learner_lr_mimic.preprocess_data()
s_learner_lr_mimic.fake_split()
s_learner_lr_mimic.external_scale_impute()
s_learner_lr_mimic.external_feature_selection()
s_learner_lr_mimic.load_model()
s_learner_lr_mimic.evaluate()
s_learner_lr_mimic.create_plot()
s_learner_lr_mimic.ite_estimation()
s_learner_lr_mimic.interpret()

# 2. s_learner_xgb_mimic
s_learner_xgb_mimic = evalpl.EvaluationPipeLine(
    name='s-learner_mimic', 
    model_source=(ROOT + '\\saved_models\\amsterdamumcdb_s-learner_XGB_model.pkl'),
    source='mimic',
    algorithm='s-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\s-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
s_learner_xgb_mimic.extract_data()
s_learner_xgb_mimic.preprocess_data()
s_learner_xgb_mimic.fake_split()
s_learner_xgb_mimic.external_scale_impute()
s_learner_xgb_mimic.external_feature_selection()
s_learner_xgb_mimic.load_model()
s_learner_xgb_mimic.evaluate()
s_learner_xgb_mimic.create_plot()
s_learner_xgb_mimic.ite_estimation()
s_learner_xgb_mimic.interpret()

# 3. s_learner_mlp_mimic
s_learner_mlp_mimic = evalpl.EvaluationPipeLine(
    name='s-learner_mimic', 
    model_source=(ROOT + '\\saved_models\\amsterdamumcdb_s-learner_MLP_model.pkl'),
    source='mimic',
    algorithm='s-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\s-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
s_learner_mlp_mimic.extract_data()
s_learner_mlp_mimic.preprocess_data()
s_learner_mlp_mimic.fake_split()
s_learner_mlp_mimic.external_scale_impute()
s_learner_mlp_mimic.external_feature_selection()
s_learner_mlp_mimic.load_model()
s_learner_mlp_mimic.evaluate()
s_learner_mlp_mimic.create_plot()
s_learner_mlp_mimic.ite_estimation()
s_learner_mlp_mimic.interpret()

# 4. t_learner_lr_mimic
t_learner_lr_mimic = evalpl.EvaluationPipeLine(
    name='t-learner_mimic', 
    m0_source=(ROOT + '\\saved_models\\amsterdamumcdb_t-learner_LR_m0_model.pkl'),
    m1_source=(ROOT + '\\saved_models\\amsterdamumcdb_t-learner_LR_m1_model.pkl'),
    source='mimic',
    algorithm='t-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\t-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
t_learner_lr_mimic.extract_data()
t_learner_lr_mimic.preprocess_data()
t_learner_lr_mimic.fake_split()
t_learner_lr_mimic.external_scale_impute()
t_learner_lr_mimic.external_feature_selection()
t_learner_lr_mimic.load_model()
t_learner_lr_mimic.evaluate()
t_learner_lr_mimic.create_plot()
t_learner_lr_mimic.ite_estimation()
t_learner_lr_mimic.interpret()

# 5. t_learner_xgb_mimic
t_learner_xgb_mimic = evalpl.EvaluationPipeLine(
    name='t-learner_mimic', 
    m0_source=(ROOT + '\\saved_models\\amsterdamumcdb_t-learner_XGB_m0_model.pkl'),
    m1_source=(ROOT + '\\saved_models\\amsterdamumcdb_t-learner_XGB_m1_model.pkl'),
    source='mimic',
    algorithm='t-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\t-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
t_learner_xgb_mimic.extract_data()
t_learner_xgb_mimic.preprocess_data()
t_learner_xgb_mimic.fake_split()
t_learner_xgb_mimic.external_scale_impute()
t_learner_xgb_mimic.external_feature_selection()
t_learner_xgb_mimic.load_model()
t_learner_xgb_mimic.evaluate()
t_learner_xgb_mimic.create_plot()
t_learner_xgb_mimic.ite_estimation()
t_learner_xgb_mimic.interpret()

# 6. t_learner_mlp_mimic
t_learner_mlp_mimic = evalpl.EvaluationPipeLine(
    name='t-learner_mimic', 
    m0_source=(ROOT + '\\saved_models\\amsterdamumcdb_t-learner_MLP_m0_model.pkl'),
    m1_source=(ROOT + '\\saved_models\\amsterdamumcdb_t-learner_MLP_m1_model.pkl'),
    source='mimic',
    algorithm='t-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\t-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
t_learner_mlp_mimic.extract_data()
t_learner_mlp_mimic.preprocess_data()
t_learner_mlp_mimic.fake_split()
t_learner_mlp_mimic.external_scale_impute()
t_learner_mlp_mimic.external_feature_selection()
t_learner_mlp_mimic.load_model()
t_learner_mlp_mimic.evaluate()
t_learner_mlp_mimic.create_plot()
t_learner_mlp_mimic.ite_estimation()
t_learner_mlp_mimic.interpret()

# 7. x_learner_lr_mimic
x_learner_lr_mimic = evalpl.EvaluationPipeLine(
    name='x-learner_mimic', 
    m0_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_LR_m0_model.pkl'),
    m1_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_LR_m1_model.pkl'),
    mx0_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_LR_mx0_model.pkl'),
    mx1_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_LR_mx1_model.pkl'),
    g_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_LR_g.pkl'),
    source='mimic',
    algorithm='x-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\t-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
x_learner_lr_mimic.extract_data()
x_learner_lr_mimic.preprocess_data()
x_learner_lr_mimic.fake_split()
x_learner_lr_mimic.external_scale_impute()
x_learner_lr_mimic.external_feature_selection()
x_learner_lr_mimic.load_model()
x_learner_lr_mimic.evaluate()
x_learner_lr_mimic.create_plot()
x_learner_lr_mimic.ite_estimation()
x_learner_lr_mimic.interpret()

# 8. x_learner_xgb_mimic
x_learner_xgb_mimic = evalpl.EvaluationPipeLine(
    name='x-learner_mimic', 
    m0_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_XGB_m0_model.pkl'),
    m1_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_XGB_m1_model.pkl'),
    mx0_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_XGB_mx0_model.pkl'),
    mx1_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_XGB_mx1_model.pkl'),
    g_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_XGB_g.pkl'),
    source='mimic',
    algorithm='x-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\t-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
x_learner_xgb_mimic.extract_data()
x_learner_xgb_mimic.preprocess_data()
x_learner_xgb_mimic.fake_split()
x_learner_xgb_mimic.external_scale_impute()
x_learner_xgb_mimic.external_feature_selection()
x_learner_xgb_mimic.load_model()
x_learner_xgb_mimic.evaluate()
x_learner_xgb_mimic.create_plot()
x_learner_xgb_mimic.ite_estimation()
x_learner_xgb_mimic.interpret()

# 9. x_learner_mlp_mimic
x_learner_mlp_mimic = evalpl.EvaluationPipeLine(
    name='x-learner_mimic', 
    m0_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_MLP_m0_model.pkl'),
    m1_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_MLP_m1_model.pkl'),
    mx0_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_MLP_mx0_model.pkl'),
    mx1_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_MLP_mx1_model.pkl'),
    g_source=(ROOT + '\\saved_models\\amsterdamumcdb_x-learner_MLP_g.pkl'),
    source='mimic',
    algorithm='x-learner',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=ROOT + '\\saved_models\\t-learner_aumcdb_label_x_selected.pkl',
    random_state=0
    )
x_learner_mlp_mimic.extract_data()
x_learner_mlp_mimic.preprocess_data()
x_learner_mlp_mimic.fake_split()
x_learner_mlp_mimic.external_scale_impute()
x_learner_mlp_mimic.external_feature_selection()
x_learner_mlp_mimic.load_model()
x_learner_mlp_mimic.evaluate()
x_learner_mlp_mimic.create_plot()
x_learner_mlp_mimic.ite_estimation()
x_learner_mlp_mimic.interpret()


# Tarnet of aumcdb on mimic
tarnet_aumcdb_on_mimic = evalpl.EvaluationPipeLine(
    name='tarnet_aumcdb_on_mimic',
    model_source=(ROOT + '\\saved_models\\amsterdamumcdb_tarnet-imacfr_model.pkl'),
    algorithm='tarnet-imacfr',
    model_scaler_source=(ROOT + '\\saved_models\\amsterdamumcdb_scaler.pkl'),
    model_imputer_source=(ROOT + '\\saved_models\\amsterdamumcdb_imputer.pkl'),
    selected_features_source=(ROOT + '\\saved_models\\tarnet-imacfr_label_x_selected.pkl'),
    source='mimic',
    random_state=42
)
tarnet_aumcdb_on_mimic.extract_data()
tarnet_aumcdb_on_mimic.preprocess_data()
tarnet_aumcdb_on_mimic.fake_split()
tarnet_aumcdb_on_mimic.external_scale_impute()
tarnet_aumcdb_on_mimic.external_feature_selection()
tarnet_aumcdb_on_mimic.load_model()
tarnet_aumcdb_on_mimic.evaluate()
tarnet_aumcdb_on_mimic.create_plot()
tarnet_aumcdb_on_mimic.ite_estimation()
tarnet_aumcdb_on_mimic.interpret()

# combine all four plots in one plot
get_overall_plot(pipeline1=imaf_pipeline, pipeline2=tarnet_aumcdb_on_mimic)

get_summary_results(
    s_learner_lr_aumcdb=s_learner_lr_aumcdb,
    s_learner_lr_mimic=s_learner_lr_mimic,
    s_learner_xgb_aumcdb=s_learner_xgb_aumcdb,
    s_learner_xgb_mimic=s_learner_xgb_mimic,
    s_learner_mlp_aumcdb=s_learner_mlp_aumcdb,
    s_learner_mlp_mimic=s_learner_mlp_mimic,
    t_learner_lr_aumcdb=t_learner_lr_aumcdb,
    t_learner_lr_mimic=t_learner_lr_mimic,
    t_learner_xgb_aumcdb=t_learner_xgb_aumcdb,
    t_learner_xgb_mimic=t_learner_xgb_mimic,
    t_learner_mlp_aumcdb=t_learner_mlp_aumcdb,
    t_learner_mlp_mimic=t_learner_mlp_mimic,
    x_learner_lr_aumcdb=x_learner_lr_aumcdb,
    x_learner_lr_mimic=x_learner_lr_mimic,
    x_learner_xgb_aumcdb=x_learner_xgb_aumcdb,
    x_learner_xgb_mimic=x_learner_xgb_mimic,
    x_learner_mlp_aumcdb=x_learner_mlp_aumcdb,
    x_learner_mlp_mimic=x_learner_mlp_mimic
)

logger.info('entire script ran succesfully')