from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import cross_val_score

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = AdaBoostRegressor(RandomForestRegressor(n_estimators=500, max_depth=78, max_features=10), n_estimators=40)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
    
