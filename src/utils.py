import os
from src.exception import CustomException
import sys
import dill


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as ce:
        raise CustomException(error_message=ce, error_detail=sys)