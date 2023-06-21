# Python Libraries
import numpy as np
import pandas as pd

# import warnings
# warnings.filterwarnings('ignore')
class Value_Imputation:
    def cleanup_old(self, subset: pd.DataFrame):
        '''replace 0s with NaN to save a two steps down the line'''
        subset.loc[:,'Value'] = subset['Value'].replace(0, np.nan)

        '''drop duplicate datetimes for each patient'''
        subset = subset.drop_duplicates(subset='GlucoseDisplayTime', keep="first")

        # ogNon0Values = subset['Value'].replace(0, np.nan).count()
        # print("starting with", ogNon0Values, "nonzero glucose measurements ('Value')")
        # print(subset.info())    

        '''Find Unrecorded 5-minute Sample'''
        patID = subset['PatientId'].iloc[0]
        
        '''set datetime as index in order for resampling method below to work'''
        subset = subset.set_index('GlucoseDisplayTime')
        
        '''fill in missing timestamps'''
        subset = subset.resample('5min').first()

        '''fix columns that *need* to be filled in'''
        subset['PatientId'] = subset['PatientId'].replace(np.nan, patID)


        '''Interpolate Missing Data'''
        # Description goes here
        # 
        """
        subset['Value'] = subset['Value'].interpolate(method='pchip')
        missing_vals = subset[subset['missing'] == 1].index
        for i in missing_vals:
            lower_t = i - timedelta(hours = 5)
            upper_t = i - timedelta(hours = 3)
            std_prev = np.sqrt(np.std(subset.loc[lower_t:upper_t, 'Value']))
            jiggle = std_prev*np.random.randn()
            subset.loc[i, 'Value'] = subset.loc[i, 'Value'] + jiggle
        """
        temp = subset['Value'].mean()
        # subset[[subset['Value'].isna(), 'Value']] = temp
        subset['Value'] = subset['Value'].fillna(temp)
        
        # print(subset.columns)
        
        # .loc[row_indexer,col_indexer] = value
        
        subset=subset.reset_index(drop=False)
        
        subset=subset[['GlucoseDisplayTime', 'PatientId', 'Value']]
        
        return subset