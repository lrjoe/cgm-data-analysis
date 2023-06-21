import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pyspark.sql.functions import pandas_udf, PandasUDFType

from datetime import date, datetime, timedelta
import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, TimestampType, IntegerType, FloatType, StringType, DateType,LongType

class Date_And_Value_Imputation:
    
    def __init__(self):
        self.spark = SparkSession.builder.appName("Glucose").getOrCreate()
        self.schema = StructType([StructField('GlucoseDisplayTime', TimestampType(),True),
                                 StructField('NumId', LongType(),True),
                                 StructField('Value', FloatType(),True),
                                 StructField('GlucoseDisplayDate', DateType(),True)])

        self.output_schema =  StructType([StructField('GlucoseDisplayTime', TimestampType(),True),
                                                     StructField('NumId', IntegerType(),True),
                                                     StructField('Value', FloatType(),True)])

    def read_interpolation(self, data_location):
        #read in data
        pyspark_glucose_data = self.spark.read \
                               .format('parquet') \
                               .load(data_location)
        
        # reorder
        pyspark_glucose_data=pyspark_glucose_data.orderBy("NumId",
                                                          "GlucoseDisplayTime",
                                                          ascending=True)
        # drop the unnecessary cols
        pyspark_glucose_data=pyspark_glucose_data.drop('__index_level_0__')

        return pyspark_glucose_data

    def interpolation_creation(self, data_set_name):
        # create data path folder
        data_location = "/cephfs/train_test_val/" + str(data_set_name)

        #get all paths in folder
        allPaths = [str(x) for x in list(pathlib.Path(data_location).glob('*.parquet')) if 'part-00' in str(x)]

        path_counter = 0
        
        # go through each path and interpolate
        for path in allPaths:
            # get data
            gluc = pd.read_parquet(path, columns=['NumId','GlucoseDisplayTime', 'Value', 'GlucoseDisplayDate'])
            # round glucoseDisplayTime to minute
            gluc['GlucoseDisplayTime'] = gluc['GlucoseDisplayTime'].dt.floor('Min')
            gluc = gluc.sort_values(by=['NumId', 'GlucoseDisplayTime'])

            # get min and max time values to only interpolate between the extremes
            min_max = gluc.groupby('NumId').agg({'GlucoseDisplayTime' : ['min','max']})

            # defaults
            merge_df = pd.DataFrame(columns=['GlucoseDisplayTime', 'NumId'])
            starttime = time.time()
            last_idx = len(min_max)-1

            index_counter = 0
            
            # iterate through each numid with their min_maxes
            for idx, row in min_max.iterrows():
                
                #grab all potential dates in range
                min_val = row['GlucoseDisplayTime']['min']
                max_val = row['GlucoseDisplayTime']['max']

                date_df = pd.DataFrame(pd.date_range(min_val, max_val, freq='5min'),\
                                   columns=['GlucoseDisplayTime'])  

                # get rows with NumId and merge in with new date_df
                # if value is na, then its the new date that needs to be interpolated
                id_df = gluc[gluc['NumId'] == idx]

                mean = id_df.Value.mean()

                id_df.set_index('GlucoseDisplayTime', inplace=True)    

                date_df.set_index('GlucoseDisplayTime', inplace=True)

                merged = id_df.join(date_df, how='outer',\
                                on='GlucoseDisplayTime', sort=True)

                # if value is na, fill IsFilledIn with 1, value = mean
                # set glucoseDisplayDate for all filled in columns
                merged['IsFilledIn'] = 0
                merged.loc[merged.Value.isna(), 'IsFilledIn'] = 1        
                merged.loc[merged.Value.isna(), 'Value'] = mean
                merged.loc[merged.GlucoseDisplayDate.isna(), 'GlucoseDisplayDate'] = merged.loc[merged.GlucoseDisplayDate.isna()]['GlucoseDisplayTime'].dt.date

                merged['NumId'] = idx

                merged.reset_index(inplace=True)

                merged = merged.drop(columns=['index'])

                # create new column that is offset time by 1
                merged['TimeLag'] = np.concatenate((merged['GlucoseDisplayTime'].iloc[0],\
                                                np.array(merged['GlucoseDisplayTime'].iloc[:-1].values)), axis=None)\
                                .astype('datetime64[ns]')
                
                # get difference between current time and time before
                merged['Diff'] = (merged['TimeLag'] - merged['GlucoseDisplayTime']).dt.seconds

                len_merged = len(merged)

                # get all index of rows with diff less than 5 mins, add 1 to remove next row, 
                # dont include last row to delete
                # remove all odd/offset time values
                indexes_to_remove = [x for x in merged[merged['Diff'] < 300].index + 1 if x < len_merged & x != 0]

                if len(indexes_to_remove) > 0:
                    merged = merged.drop(indexes_to_remove)

                merged = merged.drop(columns=['TimeLag', 'Diff'])

                # merge 25 numIds together, then once 25, save to parquet
                if ((index_counter % 25 != 0) and index_counter != last_idx) or (index_counter == 0):
                    merge_df = pd.concat([merge_df, merged])
                elif (index_counter % 25 == 0) or (index_counter == last_idx):
                    merge_df = merge_df.astype({'GlucoseDisplayTime': 'datetime64[ns]'})

                    merge_df.to_parquet('/cephfs/interpolation/val/parquet_' + str(path_counter) + '_' + str(index_counter) + '.parquet')
                    merge_df = pd.DataFrame(columns=['GlucoseDisplayTime', 'NumId'])

                index_counter += 1

            path_counter += 1

        return None
        

    def pyspark_custom_imputation_pipeline(self, df, output_schema, analysis_group):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId', 'Value', 'GlucoseDisplayTime']]

            # Imputation
            custom_imputation=Pipeline(steps=[("custom_imputation",
                                       FunctionTransformer(self.value_imputation.cleanup_old))])

            transformed_data1=custom_imputation.fit_transform(df)
            transformed_data_df=pd.DataFrame(transformed_data1)

            return transformed_data_df

        transformed_data=df.groupby(analysis_group).apply(transform_features)

        return transformed_data        

        
    def replace_missing(self, subset, patient_str):
        """ INPUT
            subset:     spark DataFrame with 1 patient
            OUTPUT
            filled_df:  spark DataFrame with 1 patient (-1 columns) and all missing rows filled in; not sorted
        """
        
        '''get first and last date (takes about 10 seconds per ten days of one patient)'''
        minimum = subset.agg({'GlucoseDisplayTime': 'min'}).collect()[0][0]
        maximum = subset.agg({'GlucoseDisplayTime': 'max'}).collect()[0][0]
        
        '''make a range that fills all those in'''
        def date_range_list(start_date, end_date):
            if start_date > end_date:
                raise ValueError("start_date must come before end_date")

            datetime_list = []
            curr_date = start_date
            while curr_date <= end_date:
                datetime_list.append([curr_date])
                curr_date += timedelta(minutes=5)
            return datetime_list
        
        datetime_list = date_range_list(minimum, maximum)
        
        '''make a dataframe of those dates'''
        deptSchema = StructType([       
            StructField('GlucoseDisplayTime', TimestampType(), True)
        ])
        dt_df = self.spark.createDataFrame(data=datetime_list, schema=deptSchema)
        
        '''merge og dataframe back into the new one'''
        merged = subset.unionByName(dt_df, allowMissingColumns=True)
        
        '''get rid of the timestamps we already have (using the exact same method as from "drop duplicate datetimes for each patient" above)'''
        window = Window.partitionBy('GlucoseDisplayTime').orderBy('tiebreak')
        merged = (merged
         .withColumn('tiebreak', monotonically_increasing_id())
         .withColumn('rank', rank().over(window))
         .filter(col('rank') == 1).drop('rank','tiebreak')
        )
        
        '''filling out the columns as needed:
            -PatientId should be all the same string
            -GlucoseDisplayTimeRaw should be used for checking the dates here, but implementation will have to come later'''
        merged = merged.fillna(patient_str, subset='PatientId')
        merged = merged.drop('GlucoseDisplayTimeRaw') #someday i'll have time to use this as the double-checker
        merged = merged.withColumn('GlucoseDisplayDate',
                                   to_date(col('GlucoseDisplayTime')))
        
        """ ============== FILL IN MISSING VALUES ============== """
        # filler = subset.agg({'Value': 'median'}).collect()[0][0]
        filler = subset.agg({'Value': 'avg'}).collect()[0][0]

        filled_df = merged.fillna(filler, subset='Value')
        
        return filled_df
    
    
    
    
    def impute_data(self, df):
        # df = self.cleanup(df)
        
        patientIds = [i.NumId for i in df.select('NumId').distinct().collect()]
        
        for ids in patientIds:
            # newData = #filter out only patient ids
            subset = df.filter("NumId = '" + str(ids) + "'")
            
            # apply funciton to get a df of the new data
            subset = self.replace_missing(subset,ids)
            
            # append that to the original df
            try:
                df_new = df_new.union(subset)
            except:
                try:
                    df_new = subset
                except:
                    raise ValueError("subsets of patient dataframes unable to append")
            
            # OR do a group by and an agg fxn
            # to do a custom agg fxn 
        
        self.spark.stop()
        
        return df_new