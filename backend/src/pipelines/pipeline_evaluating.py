"""
Программа: Полный пайплайн получения предсказаний на основе обученной модели
Версия: 1.0
"""
import pandas as pd
import yaml
import joblib
import os
import json


def pipeline_evaluate(
    config_path: str,
    df: pd.DataFrame,
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param config_path: путь до конфигурационного файла
    :param df: датасет, содержащий один или несколько объектов
    :return: предсказания
    """
    # Загрузка конфигурационного файла
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    train_config = config['training']

    # Получение датасета по заданному пути
    model = joblib.load(os.path.join(train_config['model_path']))

    # Получение предсказаний
    prediction = model.predict(df).tolist()

    with open(config['prediction']['prediction_path'], "w") as file:
        json.dump(prediction, file)

    return prediction
