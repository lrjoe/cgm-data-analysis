import plotly.express as px
import pandas as pd

class XGBoost_Classification_Plot:
    def feature_overall_importance_plot(self, feature_importance_df, overall_importance_plot_location):
        feature_importance_df=feature_importance_df.sort_values(by='Accuracy Gain', ascending=True)
        fig = px.bar(feature_importance_df, 
                     x='Accuracy Gain', 
                     y='Feature', 
                     orientation='h', 
                     color='Accuracy Gain',
                     height=1000,
                     width=900,
                     color_continuous_scale='YlGn')
        fig.write_image(overall_importance_plot_location)
        
        return fig