"""
Программа: Два верхнеуровневых метода, которые могут вызываться из frontend-части
Версия: 1.0
"""

import warnings
import optuna
import pandas as pd

from backend.src.pipelines.pipeline_training import pipeline_training
from backend.src.pipelines.pipeline_evaluating import pipeline_evaluate
from backend.src.train.metrics import load_metrics

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
CONFIG_PATH = '../config/params.yml'


def training(df: pd.DataFrame,
             empty_df=True):
    """
    Обучение модели
    """
    pipeline_training(config_path=CONFIG_PATH, df=df, empty_df=empty_df)
    metrics = load_metrics(config_path=CONFIG_PATH)
    return metrics


def prediction(df):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(
        config_path=CONFIG_PATH,
        df=df
    )

    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result}
