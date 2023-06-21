import pathlib
import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import when, col, to_date, date_trunc, rank, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DateType, FloatType, LongType

class Spark_Session:
    def __init__(self):
        self.conf = pyspark.SparkConf().setAll([\
            ('spark.app.name', 'Glucose_Analysis_Spark')])
        self.spark = SparkSession.builder.config(conf=self.conf)\
            .getOrCreate()        

class Create_Parquet_Files:
    def __init__(self):
        """Set up structs"""
        self.cohortSchema = StructType([StructField('', IntegerType(), True),
                                        StructField('UserId', StringType(), True),
                                        StructField('Gender', StringType(), True),
                                        StructField('DOB', TimestampType(), True),
                                        StructField('Age', IntegerType(), True),
                                        StructField('DiabetesType', StringType(), True),
                                        StructField('Treatment', StringType(), True)])
        self.raw_schema = StructType([StructField('_c0', IntegerType(),True),
                                      StructField('PostDate', TimestampType(),True),
                                      StructField('IngestionDate', TimestampType(),True),
                                      StructField('PostId', StringType(),True),
                                      StructField('PostTime', TimestampType(), True),
                                      StructField('PatientId', StringType(), True),
                                      StructField('Stream', StringType(), True),
                                      StructField('SequenceNumber', StringType(), True),
                                      StructField('TransmitterNumber', StringType(), True),
                                      StructField('ReceiverNumber', StringType(), True),
                                      StructField('RecordedSystemTime', TimestampType(), True),
                                      StructField('RecordedDisplayTime', TimestampType(), True),
                                      StructField('RecordedDisplayTimeRaw', TimestampType(), True),
                                      StructField('TransmitterId', StringType(), True),
                                      StructField('TransmitterTime', StringType(), True),
                                      StructField('GlucoseSystemTime', TimestampType(), True),
                                      StructField('GlucoseDisplayTime', TimestampType(), True),
                                      StructField('GlucoseDisplayTimeRaw', StringType(), True),
                                      StructField('Value', FloatType(), True),
                                      StructField('Status', StringType(), True),
                                      StructField('TrendArrow', StringType(), True),
                                      StructField('TrendRate', FloatType(), True),
                                      StructField('IsBackFilled', StringType(), True),
                                      StructField('InternalStatus', StringType(), True),
                                      StructField('SessionStartTime', StringType(), True)])
        self.step1_schema = StructType([StructField('PatientId', StringType(), True),
                                        StructField('Value', FloatType(), True),
                                        StructField('GlucoseDisplayTime', TimestampType(), True)])
        
    
    def train_val_test_step1(self,
                             csv_files_location: str="/cephfs/data",
                             checkpoint_location: str="/cephfs/train_test_val/_checkpoint.parquet"):
        spark = Spark_Session().spark

        """READ IN: all CSVs of the raw data"""
        allPaths = [str(x) for x in list(pathlib.Path(csv_files_location).glob("*.csv")) if "glucose_records" in str(x)]
        allPaths.sort()
        df = spark.read\
                  .format("csv")\
                  .option("delimiter", ',')\
                  .option("mode", "DROPMALFORMED")\
                  .option("header", True)\
                  .schema(self.raw_schema)\
                  .load(allPaths)\
                  .select(col('PatientId'), col('Value'), col('GlucoseDisplayTime'))

        """CLEAN UP"""
        # # get rid of any dates from before the actual start-date of Feb 1, 2022
        # df = df.where("GlucoseDisplayTime > '2022-01-31 23:59:59'")

        # replace 0s with NaN and dropna
        df = df.where(df.Value>0)
        df = df.na.drop(subset=['PatientId','Value','GlucoseDisplayTime'])

        df = df.withColumn("GlucoseDisplayTime",
                           date_trunc("minute",
                           col("GlucoseDisplayTime")))

        # drop duplicate datetimes for each patient
        # window = Window.partitionBy('GlucoseDisplayTime','PatientId').orderBy('tiebreak')
        # df = df.withColumn('tiebreak', monotonically_increasing_id()) \
        #        .withColumn('rank', rank().over(window)) \
        #        .filter(col('rank') == 1).drop('rank','tiebreak')

        ##### potential parquet save-out+load-in interjection here
        df.write.mode("overwrite").parquet(checkpoint_location)
        # df.write.parquet('/cephfs/train_test_val/_checkpoint.parquet')
        
        # spark.stop()
        
        
    def train_val_test_step2(self,
                            checkpoint_location: str="/cephfs/train_test_val/_checkpoint.parquet",
                            cohort_location: str="/cephfs/data/cohort.csv"):
        spark = Spark_Session().spark
        
        """RE-READ IN: all parquets from step1"""
        allPaths = [str(x) for x in list(pathlib.Path(checkpoint_location).glob("*.parquet")) if "part-0" in str(x)]
        allPaths.sort()
        df = spark.read \
                  .schema(self.step1_schema) \
                  .format("parquet") \
                  .load(allPaths)
        df = df.dropDuplicates(['PatientId', 'GlucoseDisplayTime'])
        
        """READ IN: the cohort data"""
        patientIds = spark.read \
                           .options(delimiter=',') \
                           .csv(cohort_location, header=True, schema=self.cohortSchema) \
                           .withColumnRenamed('', 'NumId') \
                           .select(col('UserId'), col('NumId')) \
                           .distinct()
        
        """Add in (join) the NumIds for future easier merges"""
        df = df.join(patientIds, df.PatientId == patientIds.UserId)\
                    .select(patientIds.NumId, df.PatientId, df.Value, df.GlucoseDisplayTime)
        
        # df.repartition('NumId').write.parquet('/cephfs/train_test_val/_checkpoint2.parquet')
        df.write.mode("overwrite").parquet(checkpoint_location)
        
    def train_val_test_step3(self,
                            checkpoint_location: str="/cephfs/train_test_val/_checkpoint2.parquet",
                            train_location: str="/cephfs/train_test_val/train_set/",
                            val_location: str="/cephfs/train_test_val/val_set/",
                            test_location: str="/cephfs/train_test_val/test_set/"):
        spark = Spark_Session().spark
        
        allPaths = [str(x) for x in list(pathlib.Path(checkpoint_location).glob('*.parquet')) if "part-0" in str(x)]
        allPaths.sort()
        
        
        for path in allPaths:
            df = spark.read.parquet(path)

            '''get total counts of values per patient'''
            counter = df.groupBy('NumId').count()
            '''get the numbers that indicate where the splits between train-val and val-test will be'''
            counter = counter.withColumn("split60",(col("count")* 0.6).cast("Integer")) \
                             .withColumn("split80",(col("count")* 0.8).cast("Integer")) \
                             .drop('count')
            '''rename for future merge (will get an "ambiguous" error without this)'''
            counter = counter.withColumnRenamed("NumId","UserId")


            '''get everything into order for ranking/sorting by 60%-20%-20%'''
            df = df.join(counter, df.NumId == counter.UserId)\
                   .select(df.NumId, df.PatientId, df.Value, df.GlucoseDisplayTime, \
                           counter.split60, counter.split80)
            df = df.orderBy("NumId", "GlucoseDisplayTime")
            '''create index (rank) column for filtering'''
            window = Window.partitionBy('NumId').orderBy('GlucoseDisplayTime')
            df = df.withColumn('rank', rank().over(window))


            """training set"""
            trainSet = df.filter((col('rank') <= col('split60')) & (col('rank') > 1)) \
                    .drop('rank','split60','split80')
            trainSet.repartition('NumId').write.mode("append").parquet(train_location)

            """validation set"""
            valSet = df.filter((col('rank') > col('split60')) & (col('rank') <= col('split80'))) \
                  .drop('rank','split60','split80')
            valSet.repartition('NumId').write.mode("append").parquet(val_location)

            """test set"""
            testSet = df.filter(col('rank') > col('split80')) \
                   .drop('rank','split60','split80')
            testSet.repartition('NumId').write.mode("append").parquet(test_location)
        
        
        
    def train_val_test_step4(self,
                             train_location: str="/cephfs/train_test_val/train_set/"):
        spark = Spark_Session().spark
        
        """RE-READ IN: all parquets from previous step"""
        allPaths = [str(x) for x in list(pathlib.Path(train_location).glob('*.parquet')) if "part-0000" in str(x)]
        allPaths.sort()
        
        for path in allPaths:
            df = spark.read.parquet(path)

            df = df.orderBy("NumId", "GlucoseDisplayTime")
            
            """create index (rank) column for filtering"""
            window = Window.partitionBy('NumId').orderBy('GlucoseDisplayTime')
            df = df.withColumn('rank', rank().over(window))

            trainSet = df.filter(col('rank') > 2) \
                    .drop('rank')
            trainSet.repartition('NumId').write.mode("append").parquet(train_location)
