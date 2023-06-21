from Input_Variables.read_vars import train_data_storage, validation_data_storage, test_data_storage, \
                                      inter_train_location, inter_test_location, inter_val_location,\
                                      one_hot_encoding_data, \
                                      analysis_group, \
                                      daily_stats_features_lower, daily_stats_features_upper, \
                                      random_seed, \
                                      create_model_bool, create_regression_bool, create_classification_bool, create_xgboost_only_bool, \
                                      xgboost_regression_model_storage_location, linear_regression_model_storage_location, \
                                      random_forest_regression_model_storage_location, factorization_machines_regression_model_storage, \
                                      xgboost_classification_model_storage_location, logistic_regression_classification_model_storage_location, \
                                      random_forest_classification_model_storage_location,final_xgboost_classification_model_storage_location, \
                                      random_seed, \
                                      final_model_load_location, \
                                      classification_evaluation_metrics_output_storage, \
                                      create_data_bool
    

from Data_Schema.schema import Pandas_UDF_Data_Schema
from Data_Generation.save_train_test_val import Create_Parquet_Files
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.imputation_pipeline import Date_And_Value_Imputation


from Feature_Generation.create_binary_labels import Create_Binary_Labels
from Feature_Generation.summary_stats import Summary_Stats_Features
from Feature_Generation.lag_features import Create_Lagged_Features
from Feature_Generation.time_series_feature_creation import TS_Features
from Feature_Generation.difference_features import Difference_Features


from Model_Preds_Eval.pyspark_model_preds_and_eval import Model_Predictions_And_Evaluations



from Model_Plots.xgboost_classification_plots import XGBoost_Classification_Plot

from Data_Pipeline.scaling_pipeline import Feature_Transformations
from Model_Creation.regression_models import Create_Regression_Models
from Model_Creation.classification_models import Create_Classification_Models
from Model_Creation.xgbclassifier_model import Create_XGBClassifier_Model

from Load_Model.load_pyspark_model import Load_And_Evaluate_Model

import os

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Glucose").config("spark.driver.memory", "4g").getOrCreate()


# PySpark UDF Schema Activation
pandas_udf_data_schema=Pandas_UDF_Data_Schema()

# Data Location
reading_data=Reading_Data()

# Create Binary y Variables
create_binary_labels=Create_Binary_Labels()

# Create and clean parquet files from CSVs
create_parquet_files = Create_Parquet_Files()

# Imputation
date_and_value_imputation=Date_And_Value_Imputation()

# Features Daily Stats Module
summary_stats_features=Summary_Stats_Features()

# Features Complex
ts_features=TS_Features()

# Features Lagged Value
create_lag_features=Create_Lagged_Features()

# Features Differences
difference_features=Difference_Features()


# Modeling Classes
feature_transformations=Feature_Transformations()
create_regression_models=Create_Regression_Models()
create_classification_models=Create_Classification_Models()
create_xgbclassifier_model=Create_XGBClassifier_Model()


# Load and evaluate feature importance
load_and_evaluate_model=Load_And_Evaluate_Model()

# Model Plots Feature Importance
xgboost_classification_plot=XGBoost_Classification_Plot()

pyspark_custom_imputation_schema=pandas_udf_data_schema.custom_imputation_pyspark_schema()


model_predictions_and_evaluations=Model_Predictions_And_Evaluations()



####### PySpark
print('Creating Data Files')
if create_data_bool is True:
    create_parquet_files.train_val_test_step1(csv_files_location="/cephfs/data",
                                              checkpoint_location="/cephfs/train_test_val/_checkpoint.parquet")
    create_parquet_files.train_val_test_step2(checkpoint_location="/cephfs/train_test_val/_checkpoint.parquet",
                                              cohort_location="/cephfs/data/cohort.csv")
    create_parquet_files.train_val_test_step3(checkpoint_location="/cephfs/train_test_val/_checkpoint.parquet",
                                             train_location="/cephfs/train_test_val/train_set/",
                                             val_location="/cephfs/train_test_val/val_set/",
                                             test_location="/cephfs/train_test_val/test_set/")
    create_parquet_files.train_val_test_step4(train_location="/cephfs/train_test_val/train_set/")


    pyspark_df=reading_data.read_in_pyspark()


    from pyspark.sql.functions import date_trunc, col

    # 
    pyspark_df=pyspark_df.withColumn("GlucoseDisplayTime", date_trunc("minute", col("GlucoseDisplayTime")))


    pyspark_df=pyspark_df.distinct()


    pyspark_df=pyspark_df.orderBy("PatientId", 
                                  "GlucoseDisplayTime",
                                  ascending=True)

###### Features Creation #######

data_types = ['train', 'test', 'val']

## data creation for train, test, and val datasets
print('Data Creation')
for dataloc in data_types:

    # check if interpolation is complete
    interpolation_complete = os.path.exists('/cephfs/interpolation/' + dataloc)
    if interpolation_complete == False:
        # if not, create interpolated data and save
        date_and_value_imputation.interpolation_creation(dataloc)


    # read in interpolation data
    custom_imputation_pipeline = date_and_value_imputation.read_interpolation('/cephfs/interpolation/' + dataloc)


    # create difference features (firstDif, secondDif)
    df_differences = difference_features.add_difference_features(custom_imputation_pipeline)
    
    # add chunk values so grouping by day can be used in complex features and summary stats
    df_chunks = summary_stats_features.create_chunk_col(df_differences, chunk_val = 288)

    #check if poincare has been performed
    poincare_complete = os.path.exists('/cephfs/featuresData/poincare/' + dataloc)
    if poincare_complete == False:
        # create poincare values and save
        df_poincare = df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.poincare)
        df_poincare.repartition('NumId').write.parquet('/cephfs/featuresData/poincare/' + dataloc)
    else:
        # if created, read in
        df_poincare = spark.read.parquet('/cephfs/featuresData/poincare/' + dataloc)

    # check if entropy is created
    entropy_complete = os.path.exists('/cephfs/featuresData/entropy/' + dataloc)
    if entropy_complete == False:
        # if not, create entropy data and save
        df_entropy = df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.entropy)
        df_entropy.repartition('NumId').write.parquet('/cephfs/featuresData/entropy/' + dataloc)
    else:
        #read in entropy data
        df_entropy = spark.read.parquet('/cephfs/featuresData/entropy/' + dataloc)

    # merge poincare and entropy together to make complex feature df
    df_complex_features = df_poincare.join(df_entropy,['NumId', 'Chunk'])

    # if summary stats have been performed
    summary_stats_complete = os.path.exists('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
    if summary_stats_complete == False:
        # create summary statistics
        features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=df_chunks)
        features_summary_stats.repartition('NumId').write.parquet('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
        
    # if summary stats exist?
    final_df_exists = os.path.exists('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
    if final_df_exists == False:
        # if not read them in
        features_summary_stats=reading_data.read_in_pyspark_data_for_summary_stats('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
        # join with complex features
        df_final = df_complex_features.join(features_summary_stats,['NumId', 'Chunk'])
        # save
        df_final.repartition('NumId').write.parquet('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')

print('Reading In clean training, validation, and test data')
training_summary = spark.read.parquet('/cephfs/summary_stats/all_train_bool_updated')
val_summary = spark.read.parquet('/cephfs/summary_stats/all_val_bool_updated')
test_summary = spark.read.parquet('/cephfs/summary_stats/all_test_bool_updated')

# Set to True in config file if you want to create the model again
if create_model_bool == True:
    print('Creating Models')
    df_train_val_combined=training_summary.union(val_summary)

    pipeline_transformation_stages=feature_transformations.numerical_scaling(df=df_train_val_combined)
    
    if create_regression_bool is True:
        regression_models_storage_locations=[xgboost_regression_model_storage_location, 
                                             linear_regression_model_storage_location,
                                             random_forest_regression_model_storage_location,
                                             factorization_machines_regression_model_storage]
        create_regression_models\
                .regression_modeling(ml_df=df_train_val_combined,
                                     stages=pipeline_transformation_stages, 
                                     random_seed=random_seed,
                                     regression_models_storage_locations=regression_models_storage_locations,
                                     num_folds=3)
        
        
    if create_classification_bool is True:
        classification_models_storage=[xgboost_classification_model_storage_location,
                                       logistic_regression_classification_model_storage_location,
                                       random_forest_classification_model_storage_location]
        create_classification_models\
                .classification_modeling(ml_df=df_train_val_combined,
                                         stages=pipeline_transformation_stages, 
                                         random_seed=random_seed,
                                         classification_models_storage_locations=classification_models_storage,
                                         num_folds=3)
        
    if create_xgboost_only_bool is True:
            xgbclassifier_model_storage=final_xgboost_classification_model_storage_location 
            create_xgbclassifier_model\
                .xgbclassifier_modeling(ml_df=df_train_val_combined,
                                        stages=pipeline_transformation_stages, 
                                        random_seed=random_seed,
                                        xgbclassifier_model_storage_location=xgbclassifier_model_storage,
                                        num_folds=3)

print('Model Evalautions')
# Model Evaluations
if create_model_bool == False:
    load_and_evaluate_model.load_evaluate_model_feature_importance(model_location=final_model_load_location)
    
    classification_models_pipeline_locations={'XGBoost': final_model_load_location}
    
    for classification_type in classification_models_pipeline_locations:
        print(f'Classification: Completing {classification_type} Model Evaluations')
        
        model_predictions_and_evaluations.classification_create_evaluations(model_type=classification_type, 
                                                                            pipeline_location=classification_models_pipeline_locations[classification_type], 
                                                                            test_data=test_summary, 
                                                                            classification_evaluation_metrics_output_storage=classification_evaluation_metrics_output_storage)