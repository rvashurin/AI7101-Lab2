import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np


BASE_PIPELINE = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
]

MODELS = {
    "simple_elastic": {
        "pipeline": Pipeline(BASE_PIPELINE + [("power", "passthrough"), ("model", ElasticNet(max_iter=5000))]),
        "param_grid": {
            "imputer__strategy": ["median", "mean"],
            "scaler": [StandardScaler(), RobustScaler()],
            "power": ["passthrough", PowerTransformer(standardize=False)],
            "model__alpha": np.logspace(-4, 2, 13).tolist(),
            "model__l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
        },
    },
    "poly_elastic": {
        "pipeline": Pipeline(
            BASE_PIPELINE
            + [
                ("power", "passthrough"),
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("model", ElasticNet(max_iter=5000)),
            ]
        ),
        "param_grid": {
            "imputer__strategy": ["median", "mean"],
            "scaler": [StandardScaler(), RobustScaler()],
            "power": ["passthrough", PowerTransformer(standardize=False)],
            "poly__degree": [2, 3],
            "poly__interaction_only": [False, True],
            "model__alpha": np.logspace(-4, 1, 10).tolist(),
            "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
    },
    "knn": {
        "pipeline": Pipeline(BASE_PIPELINE + [("power", "passthrough"), ("model", KNeighborsRegressor())]),
        "param_grid": {
            "imputer__strategy": ["median", "mean"],
            "scaler": [StandardScaler(), RobustScaler()],
            "power": ["passthrough", PowerTransformer(standardize=False)],
            "model__n_neighbors": [2, 3, 5, 8, 13, 21, 34, 55],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
            "model__leaf_size": [15, 30, 45],
        },
    },
    "svr": {
        "pipeline": Pipeline(BASE_PIPELINE + [("power", "passthrough"), ("model", SVR())]),
        "param_grid": {
            "imputer__strategy": ["median", "mean"],
            "scaler": [StandardScaler(), RobustScaler()],
            "power": ["passthrough", PowerTransformer(standardize=False)],
            "model__kernel": ["rbf", "linear"],
            "model__C": np.logspace(-2, 3, 10).tolist(),
            "model__epsilon": np.logspace(-3, 0, 8).tolist(),
            "model__gamma": ["scale", "auto"],
        },
    },
    "huber": {
        "pipeline": Pipeline(BASE_PIPELINE + [("power", "passthrough"), ("model", HuberRegressor(max_iter=5000))]),
        "param_grid": {
            "imputer__strategy": ["median", "mean"],
            "scaler": [StandardScaler(), RobustScaler()],
            "power": ["passthrough", PowerTransformer(standardize=False)],
            "model__alpha": np.logspace(-6, -1, 6).tolist(),
            "model__epsilon": [1.1, 1.35, 1.5, 1.75, 2.0],
        },
    },
    "rf": {
        "pipeline": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(random_state=0))]),
        "param_grid": {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 6, 10, 16, 24],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
    },
    "hgb": {
        "pipeline": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(random_state=0))]),
        "param_grid": {
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__max_depth": [None, 3, 5, 7],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
    },
}





def set_seed(seed: int = 1):
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(test_size: float = 0.2, random_state: int = 1):
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train(
    model: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splits: int = 5,
    random_state: int = 1,
):
    try:
        pipeline = MODELS[model]["pipeline"]
        param_grid = MODELS[model]["param_grid"]
    except KeyError:
        raise ValueError(f"Model {model} is not defined in the MODELS dictionary.")

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    return grid_search



def eval(grid_search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.DataFrame):
    best = grid_search.best_estimator_

    y_pred = best.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "estimator": best["model"],
        "preds": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
