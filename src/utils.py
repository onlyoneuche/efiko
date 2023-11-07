import os
from src.exception import CustomException
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as ce:
        raise CustomException(error_message=ce, error_detail=sys)


def evaluate_models(models, X_train, y_train, X_test, y_test, params):
    model_report = {}
    for model_name, model in models.items():

        param = params[model_name]
        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        test_score = r2_score(y_test, y_test_pred)

        model_report[model_name] = test_score
    return model_report
