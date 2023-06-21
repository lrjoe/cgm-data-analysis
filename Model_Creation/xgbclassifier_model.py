from xgboost.spark import SparkXGBClassifier

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


class Create_XGBClassifier_Model:
    def __init__(self):
        self.features_col="features"
        self.label_name="target"
        self.prediction_column_name="prediction"
            
    def xgbclassifier_modeling(self, ml_df, stages, random_seed, xgbclassifier_model_storage_location, num_folds):
        mapping= {0: 0, 1: 1, -1: 2}
        ml_df=ml_df.replace(to_replace=mapping, subset=['target'])
        
        # location_counter=0
        xgb_classification = SparkXGBClassifier(features_col=self.features_col, 
                                                label_col=self.label_name,
                                                random_state=random_seed,
                                                use_gpu=True)
        
        ml_df=ml_df.withColumn("foldCol", ml_df.NumId % num_folds)
        
        evaluator_logloss=MulticlassClassificationEvaluator(metricName='logLoss',
                                                            labelCol=self.label_name,
                                                            predictionCol=self.prediction_column_name)
        paramGrid=ParamGridBuilder().addGrid(xgb_classification.max_depth,[7]) \
                                    .addGrid(xgb_classification.n_estimators,[200,250]) \
                                    .addGrid(xgb_classification.eta,[0.4, 0.3 ,0.2]) \
                                    .addGrid(xgb_classification.reg_alpha,[12,15,]) \
                                    .addGrid(xgb_classification.reg_lambda,[0.2,0.3,0.4]) \
                                    .build()
                                    # .addGrid(xgb_classification.booster,["gbtree","dart"])\ 
                                    # .addGrid(xgb_classification.tree_method,["auto","approx"]) \
        
        # if location_counter > 0:
        #     stages.pop()
        crossval=CrossValidator(estimator=xgb_classification,
                                evaluator=evaluator_logloss,
                                estimatorParamMaps=paramGrid,
                                foldCol='foldCol',
                                collectSubModels=False)

        print('Cross Validation Occuring')
        stages.append(crossval)
        pipeline=Pipeline(stages=stages)

        model=pipeline.fit(ml_df)

        model.write().overwrite().save(xgbclassifier_model_storage_location)
        print(f'Model Saved to {xgbclassifier_model_storage_location}')
        
        return model
    
# Evaluation Metrics  
# (f1
#  accuracy
#  weightedPrecision
#  weightedRecall
#  weightedTruePositiveRate
#  weightedFalsePositiveRate
#  weightedFMeasure
#  truePositiveRateByLabel
#  falsePositiveRateByLabel
#  precisionByLabel
#  recallByLabel
#  fMeasureByLabel
#  logLoss
#  hammingLoss)