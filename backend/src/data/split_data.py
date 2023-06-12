"""
Программа: Разделение данных на train/test
Версия: 1.0
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(
        x: pd.DataFrame,
        y: pd.Series,
        shuffle: bool,
        test_size: float,
        random_state: int,
        x_test_path: str = None,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разбиение данных на обучающую и тестовую выборки
    :param x: матрица объект-признак
    :param y: целевая переменная
    :param shuffle: флаг, отвечающий за то, нужно ли перемешивать данные перед разбиением
    :param test_size: размер тестовой выборки
    :param random_state: фиксатор эксперимента
    :param x_test_path: путь для сохранения x_test
    :return: набор данных train/test
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state
    )

    if x_test_path:
        x_test.to_parquet(x_test_path)

    return x_train, x_test, y_train, y_test
