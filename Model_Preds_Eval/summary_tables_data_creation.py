print("imports start")
from Input_Variables.read_vars import train_data_storage, validation_data_storage, test_data_storage, \
                                      analysis_group, \
                                      daily_stats_features_lower, daily_stats_features_upper


from Data_Pipeline.imputation_pipeline import Date_And_Value_Imputation
from Feature_Generation.create_binary_labels import Create_Binary_Labels
from Feature_Generation.summary_stats import Summary_Stats_Features
from Feature_Generation.difference_features import Difference_Features
import os
print("imports done!")

date_and_value_imputation=Date_And_Value_Imputation()
create_binary_labels=Create_Binary_Labels()
summary_stats_features=Summary_Stats_Features()
difference_features=Difference_Features()

pipeline_stages=['train', 'val', 'test']

for pipeline_stage in pipeline_stages:
    training_files_directory=os.listdir(f'/cephfs/interpolation/{pipeline_stage}/')
    training_files=[i for i in training_files_directory if not ('.crc' in i or 'SUCCESS' in i)]

    num_training_files=len(training_files)
    training_counter=1

    for training_file in training_files:
        # Read Imputation Data
        custom_imputation_data=date_and_value_imputation.read_interpolation(f'/cephfs/interpolation/{pipeline_stage}/{training_file}')

        # Binary Labels
        added_binary_labels=create_binary_labels.pyspark_binary_labels(df=custom_imputation_data)

        # Differences Features
        differences_df=difference_features.add_difference_features(added_binary_labels)

        # Create Chunk Feature
        chunks_df=summary_stats_features.create_chunk_col(differences_df, chunk_val = 288)

        # Summary Statistics Dataframe
        features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=chunks_df)

        # Parquet File Output
        features_summary_stats.write.format('parquet').mode("overwrite").\
        save(f'/cephfs/summary_stats/{pipeline_stage}/summary_stats_{training_file}')

        # Counter Update
        print(f'Completed {pipeline_stage}: {training_counter}/{num_training_files}')
        training_counter=training_counter+1