import pandas as pd

class Feature_Importance:
    def feature_importance_accuracy_gain(self, xgboost_model, feature_importance_storage_location):
        feature_importance_dict=xgboost_model.stages[-1].bestModel.get_booster().get_score(importance_type="gain")
        feature_names=xgboost_model.stages[-2].extractParamMap()[list(xgboost_model.stages[-2].\
                                                                      extractParamMap().keys())[-1]]
        feature_importance_dict=dict(zip(feature_names, list(feature_importance_dict.values())))
        feature_importance_df=pd.DataFrame(feature_importance_dict, index=[0])

        feature_importance_df=feature_importance_df.transpose().\
                              reset_index(drop=False).\
                              rename(columns={'index': 'Feature', 0: 'Accuracy Gain'}).\
                              sort_values(by='Accuracy Gain', ascending=False).\
                              reset_index(drop=True)
        
        feature_importance_df.to_csv(feature_importance_storage_location, 
                                     index=False, 
                                     header=True)
        
        return feature_importance_df