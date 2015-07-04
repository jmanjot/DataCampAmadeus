import pandas as pd
import os

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        data_encoded = X_df
        path = os.path.dirname(__file__) 
        
        data_holidays = pd.read_csv(os.path.join(path, "data_holidays_3.csv"))
        data_holidays_encoded = data_holidays[['DateOfDeparture', 'JF-3', 'JF-2', 'JF-1', 'JF', 'JF+1', 'JF+2', 'JF+1']]
        data_encoded = data_encoded.merge(data_holidays_encoded, how='left',
        left_on=['DateOfDeparture'], 
        right_on=['DateOfDeparture'], sort=False)

        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure']+data_encoded['Arrival'], prefix='v'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
        data_encoded = data_encoded.drop('Departure', axis=1)
        data_encoded = data_encoded.drop('Arrival', axis=1)

        # following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
        data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
        # data_encoded['year'] = data_encoded['DateOfDeparture'].dt.year
        data_encoded['month'] = data_encoded['DateOfDeparture'].dt.month
        # data_encoded['day'] = data_encoded['DateOfDeparture'].dt.day
        data_encoded['weekday'] = data_encoded['DateOfDeparture'].dt.weekday
        # data_encoded['week'] = data_encoded['DateOfDeparture'].dt.week
        data_encoded['n_days'] = data_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        # data_encoded = data_encoded.join(pd.get_dummies(data_encoded['year'], prefix='y'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['month'], prefix='m'))
        # data_encoded = data_encoded.join(pd.get_dummies(data_encoded['day'], prefix='d'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['weekday'], prefix='wd'))
        # data_encoded = data_encoded.join(pd.get_dummies(data_encoded['week'], prefix='w'))

        data_encoded = data_encoded.drop('month', axis=1)
        data_encoded = data_encoded.drop('weekday', axis=1)
        data_encoded = data_encoded.drop('std_wtd', axis=1)
        data_encoded = data_encoded.drop('DateOfDeparture', axis=1)
        
        X_array = data_encoded.values
        return X_array