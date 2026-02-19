import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True),
        "KNN Classifier": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    }

def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "SVR (RBF)": SVR(kernel="rbf"),
        "KNN Regressor": KNeighborsRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    }

def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

def evaluate_regression(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }