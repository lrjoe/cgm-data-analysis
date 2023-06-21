# Python Libraries
import pandas as pd

# PySpark Libraries
import pyspark

from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import date_trunc, col, rank, when, monotonically_increasing_id
import pathlib

# Import Modules
from Data_Schema.schema import Project_Data_Schema

class Reading_Data:
    def __init__(self):
        self.project_data_schema = Project_Data_Schema()

        self.pyspark_data_schema = self.project_data_schema.data_schema_pyspark()
        self.spark = SparkSession.builder.appName("Glucose").config("spark.driver.memory", "4g").getOrCreate()
    
    
    def read_in_pyspark_data(self, data_location):
        # allPaths = [str(x) for x in list(pathlib.Path(data_location).glob('*.parquet')) if 'part-00' in str(x)]
        # allPaths.sort()
        # print(allPaths)
        
        pyspark_glucose_data = self.spark.read \
                               .schema(self.pyspark_data_schema) \
                               .format('parquet') \
                               .load(data_location)
        # pyspark_glucose_data = pyspark_glucose_data.withColumn("GlucoseDisplayTime",
        #                                                        date_trunc("minute",
        #                                                        col("GlucoseDisplayTime")))
        
        pyspark_glucose_data=pyspark_glucose_data.orderBy("PatientId",
                                                          "GlucoseDisplayTime",
                                                          ascending=True)
        
        # self.spark.stop()
        return pyspark_glucose_data

    
    def read_in_all_summary_stats(self, file_list):        
        summary_stats_df = self.spark.read \
                               .parquet(*file_list)

        return summary_stats_df
    
    
    def read_in_one_hot_encoded_data(self, one_hot_encoding_location):
        one_hot_encoding_df = self.spark.read.parquet(one_hot_encoding_location)
        
        return one_hot_encoding_df
        