from pyspark.ml import Pipeline, PipelineModel
import pandas as pd
import plotly.express as px

class Load_And_Evaluate_Model:
    def load_evaluate_model_feature_importance(self, model_location):
        # Create feature importance plot
        pipelineModel_class=PipelineModel.load(model_location)
        feature_importance_dict=pipelineModel_class.stages[-1].bestModel.get_booster().get_score(importance_type="gain")

        feature_names=pipelineModel_class.stages[-2].extractParamMap()[list(pipelineModel_class.stages[-2].\
                                                                      extractParamMap().keys())[-1]]
        feature_importance_dict=dict(zip(feature_names, list(feature_importance_dict.values())))
        feature_importance_df=pd.DataFrame(feature_importance_dict, index=[0])

        feature_importance_df=feature_importance_df.transpose().\
                              reset_index(drop=False).\
                              rename(columns={'index': 'Feature', 0: 'Accuracy Gain'}).\
                              sort_values(by='Accuracy Gain', ascending=False).\
                              reset_index(drop=True)
        
        feature_importance_df=feature_importance_df.sort_values(by='Accuracy Gain', ascending=True)
        fig = px.bar(feature_importance_df, 
                     x='Accuracy Gain', 
                     y='Feature', 
                     orientation='h', 
                     color='Accuracy Gain',
                     height=1000,
                     width=900,
                     color_continuous_scale='YlGn')
        fig.write_image('/home/jovyan/glucose-data-analysis/Output_Files/Classification/xgboost_classification_feature_importance.png')