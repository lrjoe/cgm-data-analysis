import pandas as pd
import numpy as np
import EntropyHub as eH
from pyspark.sql.functions import *
from pyspark.sql.types import *

class TS_Features:

    entropy_schema = StructType([StructField('NumId', IntegerType(), True),
                                 StructField('Chunk', IntegerType(), True),
                                 StructField('SampleEntropy', FloatType(), True),
                                 StructField('PermutationEntropy', FloatType(), True),
                                ])

    @pandas_udf(entropy_schema, functionType=PandasUDFType.GROUPED_MAP)
    def entropy(self, df):
        patientid = df['NumId'].iloc[0]
        chunk = df['Chunk'].iloc[0]
        
        glucoses = df.Value.values
        try:
            sample_entropy = eH.SampEn(glucoses, m=4)[0][-1]
            permutation_entropy = eH.PermEn(glucoses, m=4)[1][-1]
        except:
            sample_entropy, permutation_entropy = None, None
            
        entropy_df = pd.DataFrame([[patientid] + [chunk] + [sample_entropy] + [permutation_entropy]])
        entropy_df.columns=['NumId', 'Chunk', 'SampleEntropy', 'PermutationEntropy']
        return entropy_df
    
    
    poincare_schema = StructType([StructField('NumId', IntegerType(), True),
                              StructField('Chunk', IntegerType(), True),
                              StructField('ShortTermVariance', FloatType(), True),
                              StructField('LongTermVariance', FloatType(), True),
                              StructField('VarianceRatio', FloatType(), True)])
    
    @pandas_udf(poincare_schema, functionType=PandasUDFType.GROUPED_MAP)
    def poincare(self, df):
        patientid = df['NumId'].iloc[0]
        chunk = df['Chunk'].iloc[0]
        
        glucose_differentials = np.diff(df.Value)
        st_dev_differentials = np.std(np.diff(glucose_differentials))
        st_dev_values = np.std(glucose_differentials)

        # measures the width of poincare cloud
        short_term_variation = (1 / np.sqrt(2)) * st_dev_differentials

        # measures the length of the poincare cloud
        long_term_variation = np.sqrt(np.absolute((2 * st_dev_values ** 2) - (0.5 * st_dev_differentials ** 2)))
        
        if (long_term_variation != 0):
            ratio = short_term_variation / long_term_variation
        else:
            ratio = 0
        
        poincare_df = pd.DataFrame([[patientid] + [chunk] + [short_term_variation] + [long_term_variation] + [ratio]])
        poincare_df.columns=['NumId', 'Chunk', 'ShortTermVariance', 'LongTermVariance', 'VarianceRatio']
        return poincare_df
    
    
    sleep_schema = StructType([StructField('NumId', IntegerType(), True),
                              StructField('Date', DateType(), True),
                              StructField('SleepSDShort', FloatType(), True),
                              StructField('SleepSDLong', FloatType(), True),
                              StructField('SleepSDRatio', FloatType(), True)])
    
    @pandas_udf(sleep_schema, functionType=PandasUDFType.GROUPED_MAP)
    def sleep_poincare(self, df):
        patientid = df['NumId'].iloc[0]
        date = df['Date'].iloc[0]

        sleeping = df[((df.Time >= '00:00:00') & (df.Time <= '06:00:00'))]
        # sleep_values = np.array(sleeping.Value)

        sleep_differentials = np.diff(sleeping.Value)
        st_dev_differentials = np.std(np.diff(sleep_differentials))
        st_dev_values = np.std(sleep_differentials)
        
        # measures the width of poincare cloud
        short_term_variation = (1 / np.sqrt(2)) * st_dev_differentials

        # measures the length of the poincare cloud
        long_term_variation = np.sqrt(np.absolute((2 * st_dev_values ** 2) - (0.5 * st_dev_differentials ** 2)))
        
        if (long_term_variation != 0):
            ratio = short_term_variation / long_term_variation
        else:
            ratio = 0
        
        sleep_df = pd.DataFrame([[patientid] + [date] + [short_term_variation] + [long_term_variation] + [ratio]])
        sleep_df.columns=['NumId', 'Date', 'SleepSDShort', 'SleepSDLong', 'SleepSDRatio']
        return sleep_df
    
    
    def process_for_sleep_poincare(self, df):
        df = df.withColumn('Date', to_date(col('GlucoseDisplayTime')))
        df = df.withColumn('Month', month(df.Date))
        df = df.withColumn('Time', date_format('GlucoseDisplayTime', 'HH:mm:ss'))
        
        sleep_df = df.groupby(['NumId', 'Month']).apply(self.sleep)
        return sleep_df
        
    
    
    