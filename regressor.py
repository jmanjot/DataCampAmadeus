from sklearn.ensemble import RandomForestRegressor 
from sklearn.base import BaseEstimator 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.base import BaseEstimator 
from sklearn.decomposition import KernelPCA 
from sklearn import neighbors 
from sklearn.decomposition import PCA 
import xgboost as xgb 


  
class Regressor(BaseEstimator): 
    def __init__(self): 
        self.clf0 = GradientBoostingRegressor( n_estimators = 2000 , max_depth = 9 , max_features = 27) 
        self.clf1 = xgb.XGBRegressor(max_depth=17, n_estimators=1000, learning_rate=0.05) 
        self.clf2 = xgb.XGBRegressor(max_depth=17, n_estimators=1000, learning_rate=0.04) 
          
 
 
    def fit(self, X, y): 
        self.clf0.fit(X[:,0:102], y) 
        self.clf1.fit(X[:,0:102], y) 
        self.clf2.fit(X[:,102:], y) 
 
 
    def predict(self, X): 
        #list_clf=[self.clf0.predict(X[:,0:102]),self.clf1.predict(X[:,102:])] 
        #list_clf=[self.clf0.predict(X[:,0:102])] 
        #return sum(list_clf)/float(len(list_clf)) 
        return self.clf0.predict(X[:,0:102]) * 0.5 + self.clf1.predict(X[:,0:102]) * 0.3 +  self.clf2.predict(X[:,102:]) * 0.2 
 