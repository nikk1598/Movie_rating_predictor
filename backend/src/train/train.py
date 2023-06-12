"""
Программа: Обучение модели
Версия: 1.0
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import optuna
from ..data.split_data import split_train_test
from ..train.metrics import save_metrics
from sklearn.metrics import mean_squared_error


def objective_lgb(trial,
                  x_train: pd.DataFrame,
                  y_train: pd.Series,
                  x_test: pd.DataFrame,
                  y_test: pd.Series,
                  random_state: int = 10) -> np.array:
    """
    Целевая функция для поиска параметров
    :param x_train: обучающая выборка без целевой переменной
    :param x_test: тестовая выборка без целевой переменной
    :param y_train: целевая переменная в обучающей выборке
    :param y_test: целевая переменная в тестовой выборке
    :param trial: число итераций
    :param random_state: фиксатор эксперимента
    :return: среднее значение по метрике mse на данной итерации
    """
    # Определение гиперпараметров для оптимизации
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'random_state': random_state
    }

    # Создание модели с выбранными гиперпараметрами
    model = LGBMRegressor(**params)

    # Обучение модели на тренировочных данных
    model.fit(x_train, y_train)

    # Получение предсказаний на тестовых данных
    y_pred = model.predict(x_test)

    # Вычисление метрики для оценки качества модели (например, среднеквадратичная ошибка)
    mse = mean_squared_error(y_test, y_pred)

    return mse


def find_optimal_params(
        n_trials: int,
        x: pd.DataFrame,
        y: pd.Series,
        shuffle: bool,
        test_size: float,
        random_state: int = 10) -> optuna.Study:
    """
    Нахождение оптимальных гиперпараметров модели
    :param n_trials: число итераций
    :param x: матрица объект признак
    :param y: целевая переменная
    :param shuffle: флаг, отвечающий за то, нужно ли перемешивать данные перед разбиением
    :param test_size: размер тестовой выборки
    :param random_state: фиксатор эксперимента
    :return: объект класса optuna.Study, хранящий всю информацию об обучении
    """

    x_train, x_test, y_train, y_test = split_train_test(
        x=x,
        y=y,
        shuffle=shuffle,
        test_size=test_size,
        random_state=random_state,
    )

    sampler = optuna.samplers.RandomSampler(seed=10)
    study = optuna.create_study(direction="minimize", study_name="LightGBM", sampler=sampler)
    func = lambda trial: objective_lgb(
        trial,
        x_train,
        y_train,
        x_test,
        y_test,
        random_state=random_state
    )

    study.optimize(func, n_trials=n_trials, show_progress_bar=True)

    return study


def train_model(
        x: pd.DataFrame,
        y: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        study: optuna.Study,
        metric_path: str) -> LGBMRegressor:
    """
    Обучение модели на лучших параметрах
    :param x: матрица объект-признак (обучающая выборка)
    :param y: целевая переменная (обучающая выборка)
    :param x_test: матрица объект-признак (тестовая выборка)
    :param y_test: целевая переменная (тестовая выборка)
    :param study: объект класса optuna.Study, хранящий всю информацию об обучении
    :param metric_path: путь до пути с метриками
    :return: объект класса LGBMRegressor с подобранными параматрами
    """

    lgb_grid = LGBMRegressor(**study.best_params)
    lgb_grid.fit(x, y)

    save_metrics(x=x_test, y=y_test, model=lgb_grid, metric_path=metric_path)
    return lgb_grid
