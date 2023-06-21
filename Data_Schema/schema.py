from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType, DateType, IntegerType
class Project_Data_Schema:
    def data_schema_pyspark(self):        
        glucose_data_schema=StructType([StructField('NumId', IntegerType(), True),
                                        StructField('PatientId', StringType(), True),
                                        StructField('Value', FloatType(), True),
                                        StructField('GlucoseDisplayTime', TimestampType(), True),
                                        StructField('GlucoseDisplayTimeRaw', StringType(), True),
                                        StructField('GlucoseDisplayDate', DateType(), True)])
        return glucose_data_schema
    

class Pandas_UDF_Data_Schema:
    def custom_imputation_pyspark_schema(self):
        pyspark_custom_imputation_schema=StructType([StructField('GlucoseDisplayTime', TimestampType(),True),
                                                     StructField('PatientId', StringType(),True),
                                                     StructField('Value', FloatType(),True)])
        
        return pyspark_custom_imputation_schema
