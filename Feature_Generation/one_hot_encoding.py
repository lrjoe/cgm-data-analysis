import pyspark
from pyspark import pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import OneHotEncoder, StringIndexer

class OneHotEncoding:
    def add_encoding_to_patients(self):
        spark = Spark_Session().spark
        
        df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv('/cephfs/data/cohort.csv')
        df = df.withColumn('AgeGroup', substring(df.Age.cast(StringType()), 0,1))
        df = df.withColumnRenamed('Gender', 'Sex')
        
        # assign index to string vals for OneHotEncoding
        encodedCols = ['Sex', 'Treatment', 'AgeGroup'] # not doing'DiabetesType' because all type-two
        encodedLabels = []

        for name in encodedCols:
            indexer = StringIndexer(inputCol=name, outputCol= name + '_Num')
            indexer_fitted = indexer.fit(df)
            encodedLabels.append([name, indexer_fitted.labels])

            df = indexer_fitted.transform(df)
            
        #if you want to understand what each encoding label means    
        # order of index is based on frequency, most freq at beginning
        #[['Sex', ['Female', 'Male']],
        # ['Treatment', ['yes-both', 'yes-long-acting', 'no', 'yes-fast-acting']]]
        # ['AgeGroup', ['5', '6', '7', '4', '3', '8', '9', '1']]]
        single_col_ohe = OneHotEncoder(inputCol="Sex_Num", outputCol="Sex_Encoded", dropLast=True)
        df = single_col_ohe.fit(df).transform(df)

        single_col_ohe = OneHotEncoder(inputCol="Treatment_Num", outputCol="Treatment_Encoded", dropLast=True)
        df = single_col_ohe.fit(df).transform(df)

        single_col_ohe = OneHotEncoder(inputCol="AgeGroup_Num", outputCol="AgeGroup_Encoded", dropLast=True)
        df = single_col_ohe.fit(df).transform(df)
        
        df.write.parquet('/cephfs/data/cohort_encoded.parquet')