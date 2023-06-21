# Python Libraries
import numpy as np
import pandas as pd

from pyspark.sql.functions import col, to_date, sum, avg, max, min, \
stddev, percentile_approx,\
pandas_udf, PandasUDFType, lit, udf, collect_list, sqrt, monotonically_increasing_id, map_from_entries,\
rank, dense_rank, count, when, lag

from pyspark.sql.types import IntegerType

from pyspark.sql.window import Window

class Summary_Stats_Features:
    def create_chunk_col(self, df, chunk_val):
        # create a chunk column that is the index/chunk_val 
        # changing it to int rounds down. this creates groups of chunk_val length
        window = Window.partitionBy(df['NumId']).orderBy(df['GlucoseDisplayTime'])
        df = df.select('*', rank().over(window).alias('index'))
        df = df.withColumn("Chunk", (df.index/chunk_val).cast(IntegerType()))

        return df


    def pyspark_summary_statistics(self,
                                   df, \
                                   chunk_val = 288, 
                                   chunk_lag=1):  

        group_cols = ["NumId", "Chunk"]

        # create agg summary stats
        summary_df = df.groupby(group_cols)\
            .agg(avg("Value").alias("Mean"),\
                 stddev("Value").alias("StdDev"),\
                 percentile_approx("Value", .5).alias("Median"), \
                 min("Value").alias("Min"),\
                 max("Value").alias("Max"),\
                 avg('FirstDiff').alias('AvgFirstDiff'),\
                 avg('SecDiff').alias('AvgSecDiff'),\
                 stddev('FirstDiff').alias('StdFirstDiff'),\
                 stddev('SecDiff').alias('StdSecDiff'),\
                 sum(col("is_above")).alias("CountAbove"),\
                 sum(col("is_below")).alias("CountBelow"),\
                 sum(col('y_Binary')).alias('TotalOutOfRange')
                )
        
        my_window = Window.partitionBy("NumId").orderBy("Chunk")
        
        summary_df = summary_df.withColumn("NextDayValue", lag(summary_df.TotalOutOfRange, offset=chunk_lag).over(my_window))
        summary_df = summary_df.withColumn("DiffPrevious", summary_df.NextDayValue - summary_df.TotalOutOfRange)
        
        # to give patients leeway, if they are within a 45min window, set it to 0
        
        buffer = 9  # 45 minutes / 5 minute intervals = 9
        summary_df = summary_df.withColumn('target', when(summary_df.DiffPrevious > buffer, 1)
                                 .when(summary_df.DiffPrevious < buffer, -1)
                                 .otherwise(0))
        
        summary_df = summary_df.drop('NextDayValue')
        
        summary_df=summary_df.filter(summary_df.target.isNotNull())
        
        return summary_df