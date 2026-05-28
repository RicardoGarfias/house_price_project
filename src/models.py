import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple


def train_regression_models(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, Ridge, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    ridge = Ridge(alpha=1.0)
    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    results = {}
    for name, model in [('Random Forest', rf), ('Ridge Regression', ridge)]:
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared=False),
            'R2': r2_score(y_test, y_pred)
        }

    return rf, ridge, results
