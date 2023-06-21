import pyspark.sql.functions as F
from pyspark.sql.window import Window

class Create_Lagged_Features:
    def pyspark_lag_features(self, df, time_series_lag_values_created):
        w=Window.partitionBy('PatientId').orderBy('GlucoseDisplayTime')
        max_time_lag=time_series_lag_values_created+1

        for i in range(1, max_time_lag): 
            df=df.withColumn(f"value_lag_{i}", F.lag(F.col('Value'), i).over(w))
            df=df.withColumn(f"mean_lag_{i}", F.lag(F.col('Mean'), i).over(w))
            df=df.withColumn(f"std_dev_lag_{i}", F.lag(F.col('Std Dev'), i).over(w))
            df=df.withColumn(f"med_lag_{i}", F.lag(F.col('Median'), i).over(w))
            df=df.withColumn(f"min_lag_{i}", F.lag(F.col('Min'), i).over(w))
            df=df.withColumn(f"max_lag_{i}", F.lag(F.col('Max'), i).over(w)) 
            df=df.withColumn(f"cnt_bel_lag_{i}", F.lag(F.col('CountBelow'), i).over(w))
            df=df.withColumn(f"cnt_abv_lag_{i}", F.lag(F.col('CountAbove'), i).over(w))
            df=df.withColumn(f"perc_belw_lag_{i}", F.lag(F.col('PercentageBelow'), i).over(w))
            df=df.withColumn(f"perc_abv_lag_{i}", F.lag(F.col('PercentageAbove'), i).over(w))

        df=df.na.drop()

        return df