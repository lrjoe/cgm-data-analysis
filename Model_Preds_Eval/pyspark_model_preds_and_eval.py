from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pandas as pd

class Model_Predictions_And_Evaluations:
    def regression_create_predictions(self, test_df, model):
        predict_df=model.transform(test_df).select('NumId', 'Chunk', 'prediction', 'DiffPrevious')
        
        return predict_df
    
    def regression_create_evaluations(self, model_type, pipeline_location, test_data, regression_evaluation_metrics_output_storage):
        pipeline_model=PipelineModel.load(pipeline_location)
        testing_predictions=self.regression_create_predictions(test_df=test_data, 
                                                               model=pipeline_model)
        
        evaluators=['rmse', 'mse', 'r2', 'mae', 'var']
        metrics_dict={'rmse': None,
                      'mse': None,
                      'r2': None,
                      'mae': None,
                      'var': None}
        
        for metric in evaluators:
            eval_metric=RegressionEvaluator(labelCol="DiffPrevious", 
                                            predictionCol="prediction", 
                                            metricName=metric)
            metric_value=eval_metric.evaluate(testing_predictions)
            metrics_dict[metric]=metric_value
            
        
        eval_df=pd.DataFrame(metrics_dict, index=[0])
        
        output_location=regression_evaluation_metrics_output_storage+model_type+'eval_metrics.csv'
        eval_df.to_csv(output_location, index=False, header=True)
        
        
        
        
    def classification_create_predictions(self, test_df, model):
        predict_df=model.transform(test_df).select('NumId', 'Chunk', 'rawPrediction', 'probability', 'prediction', 'target')
        mapping= {0: 0, 1: 1, -1: 2}
        predict_df=predict_df.replace(to_replace=mapping, subset=['target'])
        
        return predict_df
    
    def classification_create_evaluations(self, model_type, pipeline_location, test_data, classification_evaluation_metrics_output_storage):
        pipeline_model=PipelineModel.load(pipeline_location)
        testing_predictions=self.classification_create_predictions(test_df=test_data, 
                                                                   model=pipeline_model)
        
        testing_predictions=testing_predictions \
            .withColumn("target", testing_predictions["target"].cast("double")) \
            .withColumn("prediction", testing_predictions["prediction"].cast("double"))
        
        
        metrics_dict={'accuracy': None,
             'precisionByLabel': None,
             'recallByLabel': None,
             'f1': None,
             'confusion_matrix': None}
        
        
        for metric in ['accuracy', 'precisionByLabel', 'recallByLabel', 'f1']:
            eval_metric=MulticlassClassificationEvaluator(labelCol="target", 
                                                          predictionCol="prediction", 
                                                          metricName=metric)
            metric_value=eval_metric.evaluate(testing_predictions)
            metrics_dict[metric]=metric_value
            
        preds_and_labels=testing_predictions.select(['prediction','target']).withColumn('label', F.col('target').cast(FloatType())).orderBy('prediction')
        preds_and_labels = preds_and_labels.select(['prediction','label'])
        metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
        cf_matrix=metrics.confusionMatrix().toArray()
        metrics_dict['confusion_matrix']=f'{cf_matrix}'

        eval_df=pd.DataFrame(metrics_dict, index=[0])
        
        output_location=classification_evaluation_metrics_output_storage+model_type+'eval_metrics.csv'
        eval_df.to_csv(output_location, index=False, header=True)