from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType, DateType, IntegerType

import numpy as np

class Create_Binary_Labels:    

    #     def pyspark_binary_labels(self, df):
#         #get 10th and 90th percentiles of patient
#         lower_10, upper_90  = df.approxQuantile('Value', [.1, .9], 0)
        
#         #if 10th percentile of patient is lower than default, use percentile
#         lower = lower_10 if lower_10 < self.lower else self.lower
#         upper = upper_90 if upper_90 > self.upper else self.upper
        
#         df=df.withColumn('y_Binary', F.when(F.col('Value') > upper, 1)\
#             .when(F.col('Value') < lower, 1)\
#                 .otherwise(0))
#         df=df.withColumn('is_above', F.when(F.col('Value') > upper, 1).otherwise(0))
#         df=df.withColumn('is_below', F.when(F.col('Value') < lower, 1).otherwise(0))

#         return df

    # outSchema = StructType([StructField('GlucoseDisplayTime', TimestampType(), True),
    #                         StructField('NumId', IntegerType(), True),
    #                         StructField('Value', FloatType(), True),
    #                         StructField('IsFilledIn', FloatType(), True),
    #                         StructField('y_Binary', IntegerType(), True),
    #                         StructField('is_above', IntegerType(), True),
    #                         StructField('is_below', IntegerType(), True)
    #                     ])
    
    # outSchema = StructType([StructField('NumId', IntegerType(), True),
    #                         StructField('PatientId', StringType(), True),
    #                         StructField('Value', FloatType(), True),
    #                         StructField('GlucoseDisplayTime', TimestampType(), True),
    #                         StructField('IsFilledIn', FloatType(), True),
    #                         StructField('y_Binary', IntegerType(), True),
    #                         StructField('is_above', IntegerType(), True),
    #                         StructField('is_below', IntegerType(), True)
    #                     ])
    
    outSchema = StructType([StructField('GlucoseDisplayTime', TimestampType(), True),
                            StructField('NumId', IntegerType(), True),
                            StructField('Value', FloatType(), True),
                            StructField('IsFilledIn', FloatType(), True),
                            StructField('y_Binary', IntegerType(), True),
                            StructField('is_above', IntegerType(), True),
                            StructField('is_below', IntegerType(), True)
                    ])
    
    @pandas_udf(outSchema, functionType=F.PandasUDFType.GROUPED_MAP)
    def pandas_binary_labels(self, df):
        lower_10, upper_90 = df.Value.quantile([.1, .9], interpolation='nearest')
        
        default_lower = 70
        default_upper = 180
        
        lower = lower_10 if lower_10 < default_lower else default_lower
        upper = upper_90 if upper_90 > default_upper else default_upper        
        df['y_Binary'] = [1 if ((x > upper) or (x < lower)) else 0 for x in df['Value']]
        df['is_above'] = [1 if (x > upper) else 0 for x in df['Value']]
        df['is_below'] = [1 if (x < lower) else 0 for x in df['Value']]

        return df  
        
    def pyspark_binary_labels(self, df):
        return df.groupBy('NumId').apply(self.pandas_binary_labels)
    