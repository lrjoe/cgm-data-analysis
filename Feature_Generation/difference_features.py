import pyspark
from pyspark.sql.functions import col, lag, when, isnull
from pyspark.sql.types import StructType, StructField, \
StringType, IntegerType, TimestampType, DateType, FloatType
from pyspark.sql.window import Window


class Difference_Features:
    def add_difference_features(self, df):
        
        my_window = Window.partitionBy('NumId').orderBy("GlucoseDisplayTime")
        # create column that is one lagged display time
        df = df.withColumn("prev_value", lag(df.Value).over(my_window))
        # first diff is the difference in value between that time and previous time
        df = df.withColumn("FirstDiff", when(isnull(df.Value - df.prev_value), 0)
                                  .otherwise(df.Value - df.prev_value))

        # redo above computation for second diff
        df = df.withColumn("prev_val_sec", lag(df.FirstDiff).over(my_window))
        df = df.withColumn("SecDiff", when(isnull(df.FirstDiff - df.prev_val_sec), 0)
                                  .otherwise(df.FirstDiff - df.prev_val_sec))

        #drop unnecessary columns
        df = df.drop('prev_value', 'prev_val_sec')
        
        return df
        
        