import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class MachineLearner:
    # 초기화
    def __init__(self, model_type='linear', **kwargs):
        if model_type == 'linear':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**kwargs)
        else:
            raise ValueError("model_type must be 'linear' or 'lightgbm'")
        self.model_type = model_type

    # 모델학습
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # 예측
    def predict(self, X_test):
        return self.model.predict(X_test)

    # 스코어링(평가)
    def score(self, X_test, y_test, metric='r2'):
        y_pred = self.predict(X_test)
        if metric == 'r2':
            return r2_score(y_test, y_pred)
        elif metric == 'mse':
            return mean_squared_error(y_test, y_pred)
        else:
            raise ValueError("metric must be 'r2' or 'mse'")

    def get_model(self):
        return self.model