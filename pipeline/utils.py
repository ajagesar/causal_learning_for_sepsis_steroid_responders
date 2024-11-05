from os.path import dirname as up
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set absulute path of project for referencing files
ROOT = up(up(__file__))

# set logger settings
#logging.basicConfig(filename=ROOT + '\\logs\\log_file.log', level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S')


# settings (to change if working on mac with MPS)
#device = torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# df to tensor
to_tensor = lambda x: torch.tensor(np.array(x), dtype = torch.float32).to(device)

# TODO tensor to df

# set data for further use
def get_preprocessed_data(self):
    if self.metalearner_classifier == 'XGB':
        data = self.preprocessed_data_scaled.copy()
    if self.metalearner_classifier != 'XGB':
        data = self.preprocessed_data_scaled_imputed.copy()
    return data

# create plot from 4 subplots
def get_overall_plot(pipeline1, pipeline2):
    figure, axis = plt.subplots(1, 2, figsize=(16,8))

    axis[0].plot(pipeline1.plt_fpr, pipeline1.plt_tpr, 'b', label='AUC AUMCDB = %0.2f' %pipeline1.plt_roc_auc)
    axis[0].plot(pipeline2.plt_fpr, pipeline2.plt_tpr, 'g', label='AUC MIMIC-IV = %0.2f' %pipeline2.plt_roc_auc)
    axis[0].set_title('Receiver Operating Characteristic')
    axis[0].set_xlabel('False Positive Rate')
    axis[0].set_ylabel('True Positive Rate')
    axis[0].legend(loc='lower right')

    axis[1].plot(pipeline1.plt_pred_pos, pipeline1.plt_true_pos, marker='o', color='b', label='AUMCDB')
    axis[1].plot(pipeline2.plt_pred_pos, pipeline2.plt_true_pos, marker='o', color='g', label='MIMIC-IV')
    axis[1].plot([0,1], [0,1], linestyle='--')
    axis[1].set_title('Calibration plot')
    axis[1].set_xlabel('Predicted probability')
    axis[1].set_ylabel('True probability')
    axis[1].legend(loc='lower right')

    figure.savefig(ROOT + f'\\saved_csvs\\auc_calibration.png')

# get summary table
def get_summary_results(
        s_learner_lr_aumcdb,
        s_learner_lr_mimic,
        s_learner_xgb_aumcdb,
        s_learner_xgb_mimic,
        s_learner_mlp_aumcdb,
        s_learner_mlp_mimic,
        t_learner_lr_aumcdb,
        t_learner_lr_mimic,
        t_learner_xgb_aumcdb,
        t_learner_xgb_mimic,
        t_learner_mlp_aumcdb,
        t_learner_mlp_mimic,
        x_learner_lr_aumcdb,
        x_learner_lr_mimic,
        x_learner_xgb_aumcdb,
        x_learner_xgb_mimic,
        x_learner_mlp_aumcdb,
        x_learner_mlp_mimic
):

    columns = [
        'Meta-learner',
        'Algorithm',
        'AUC internal test set (AumsterdamUMCdb)',
        'AUC external test set (MIMIC-IV)'
    ]

    content_dict = {
        0:['S-learner', 'Logistic Regression', round(s_learner_lr_aumcdb.auc,2), round(s_learner_lr_mimic.auc,2)], # TODO add other models
        1:['S-learner', 'XGBoost', round(s_learner_xgb_aumcdb.auc,2), round(s_learner_xgb_mimic.auc,2)],
        2:['S-learner', 'MLP', round(s_learner_mlp_aumcdb.auc,2), round(s_learner_mlp_mimic.auc,2)],
        3:['T-learner', 'Logistic Regression', round(t_learner_lr_aumcdb.auc,2), round(t_learner_lr_mimic.auc,2)],
        4:['T-learner', 'XGBoost', round(t_learner_xgb_aumcdb.auc,2), round(t_learner_xgb_mimic.auc,2)],
        5:['T-learner', 'MLP', round(t_learner_mlp_aumcdb.auc,2), round(t_learner_mlp_mimic.auc,2)],
        6:['X-learner', 'Logistic Regression', round(x_learner_lr_aumcdb.auc,2), round(x_learner_lr_mimic.auc,2)],
        7:['X-learner', 'XGBoost', round(x_learner_xgb_aumcdb.auc,2), round(x_learner_xgb_mimic.auc,2)],
        8:['X-learner', 'MLP', round(x_learner_mlp_aumcdb.auc,2), round(x_learner_mlp_mimic.auc,2)]
    }

    df = pd.DataFrame.from_dict(data=content_dict, orient='index', columns=columns)

    df.to_markdown(ROOT + '\\paper\\output\\other_models_performance.txt', index=False)
