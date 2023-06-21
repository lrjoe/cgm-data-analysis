import yaml
import numpy as np
# import fathon
# from fathon import fathonUtils as fu

with open('Input_Variables/data_vars.yaml', 'r') as file:
    input_vars=yaml.safe_load(file)


################################### Data Location ###################################
train_data_storage=input_vars['Data_Storage']['train_data_storage']
validation_data_storage=input_vars['Data_Storage']['validation_data_storage']
test_data_storage=input_vars['Data_Storage']['test_data_storage']

inter_train_location=input_vars['Data_Storage']['inter_train_location']
inter_test_location=input_vars['Data_Storage']['inter_test_location']
inter_val_location=input_vars['Data_Storage']['inter_val_location']

one_hot_encoding_data=input_vars['Data_Storage']['one_hot_encoding_location']

################################### Data Creation ###################################
create_data_bool=input_vars['Create_Data']

################################### Evaluation Metrics ###################################
regression_evaluation_metrics_output_storage=input_vars['Evaluation_Metrics']['regression_evaluation_metrics_output_storage']
classification_evaluation_metrics_output_storage=input_vars['Evaluation_Metrics']['classification_evaluation_metrics_output_storage']


################################### Feature Importance ###################################
feature_importance_storage_location=input_vars['Feature_Importance']['feature_importance_storage_location']
overall_feature_importance_plot_location=input_vars['Feature_Importance']['overall_feature_importance_plot_location']

################################### Analysis ###################################
analysis_group=input_vars['Analysis']['Analysis_Group']


################################### Time Series Input Features ###################################
mfdfa_win_1=input_vars['Time_Series_Input_Features']['MFDA']['win_1']
mfdfa_win_2=input_vars['Time_Series_Input_Features']['MFDA']['win_2']
# mfda_win_sizes=fu.linRangeByStep(mfdfa_win_1, mfdfa_win_2) # 30 mins to 1/2 year for

mfdfa_q_list_1=input_vars['Time_Series_Input_Features']['MFDA']['q_list_1']
mfdfa_q_list_2=input_vars['Time_Series_Input_Features']['MFDA']['q_list_2']
mfda_q_list=np.arange(mfdfa_q_list_1, mfdfa_q_list_2)

mfdfa_rev_seg=input_vars['Time_Series_Input_Features']['MFDA']['rev_seg']
mfdfa_pol_order=input_vars['Time_Series_Input_Features']['MFDA']['pol_order']

################################### Daily Stats Features ###################################
daily_stats_features_lower=input_vars['Daily_Stats_Features']['lower']
daily_stats_features_upper=input_vars['Daily_Stats_Features']['upper']


################################### ML Models ###################################
# Regression
xgboost_regression_model_storage_location=input_vars['ML_Models']['Regression']['xgboost_regression_model_storage_location']
linear_regression_model_storage_location=input_vars['ML_Models']['Regression']['linear_regression_model_storage_location']
random_forest_regression_model_storage_location=input_vars['ML_Models']['Regression']['random_forest_regression_model_storage_location']
factorization_machines_regression_model_storage=input_vars['ML_Models']['Regression']['factorization_machines_regression_model_storage']

# Classification
xgboost_classification_model_storage_location=input_vars['ML_Models']['Classification']['xgboost_classification_model_storage_location']
logistic_regression_classification_model_storage_location=input_vars['ML_Models']['Classification']['logistic_regression_classification_model_storage_location']
random_forest_classification_model_storage_location=input_vars['ML_Models']['Classification']['random_forest_classification_model_storage_location']

# final XGBoost Classification
final_xgboost_classification_model_storage_location=input_vars['ML_Models']['Final_Classification']['final_xgboost_classification_model_storage_location']

# Setting Seed
random_seed=input_vars['ML_Models']['random_seed']

# Create Model
create_model_bool=input_vars['ML_Models']['model_creation']['create_model']

# Do You want to create the regression model
create_regression_bool=input_vars['ML_Models']['model_creation']['train_regression']

# Do You want to create the classification model
create_classification_bool=input_vars['ML_Models']['model_creation']['train_classification']

# Do You want to create only the xgboost model
create_xgboost_only_bool=input_vars['ML_Models']['model_creation']['train_only_xgboost']

# final_model_load_location
final_model_load_location=input_vars['Load_ML_Model']['final_model_load_location']

