"""
Программа: Получение метрик
Версия: 1.0
"""

import pandas as pd
import numpy as np
import json
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def wape(y_true: list, y_pred: list) -> float:
    return np.sum(np.abs(np.array(y_pred) - np.array(y_true))) / np.sum(y_true)


def r2_adjusted(y_true: list, y_pred: list, x_test: np.array) -> float:
    n_objects = len(y_true)
    n_features = x_test.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)


def create_dict_metrics(y_test: pd.Series,
                        y_predict: pd.Series,
                        x_test: pd.DataFrame) -> dict:
    """
    Создание словаря с метриками для задачи бинарной классификации
    :param y_test: истинные метки
    :param y_predict: предсказанные метки
    :param x_test: датасет без метки
    :return: словарь с метриками
    """
    dict_metrics = {
        "mae": round(mean_absolute_error(y_test, y_predict), 3),
        "mse": round(mean_squared_error(y_test, y_predict), 3),
        "r2_adjusted": round(r2_adjusted(list(y_test), list(y_predict), x_test), 3),
        "wape": round(wape(list(y_test), list(y_predict)), 3)
    }
    return dict_metrics


def save_metrics(x: pd.DataFrame,
                 y: pd.Series,
                 model: object,
                 metric_path: str) -> None:
    """
    Сохранение словаря с метриками по заданному пути
    :param x: матрица объект-признак
    :param y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения
    :return: None
    """
    result_metrics = create_dict_metrics(
        y_test=y,
        y_predict=model.predict(x),
        x_test=x
    )

    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config['training']['metrics_path'], encoding='utf-8') as json_file:
        metrics = json.load(json_file)

    return metrics
