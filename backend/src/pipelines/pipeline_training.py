"""
Программа: Полный пайплайн обучения модели
Версия: 1.0
"""

import yaml
import joblib
import os
import datetime
import pandas as pd
from ..data.split_data import split_train_test
from ..data.get_data import get_imdb_dataset
from ..data.transform_data import transform_imdb_dataset
from ..train.train import train_model, find_optimal_params


def pipeline_training(config_path: str,
                      df: pd.DataFrame,
                      empty_df=True) -> None:
    """
    Полный цикл получения данных и тренировки модели
    :param df: датасет
    :param config_path: путь до конфигурационного файла
    :param empty_df: параметр, указывающий на то, был ли передан датасет для обучения
    :return: None
    """

    # Загрузка конфигурационного файла
    with open(config_path,  encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    train_config = config['training']
    unique_config = config['unique_data']

    if empty_df:
        # Загрузка сырого датасета
        df_from_imdb = get_imdb_dataset(config_path=config_path)
        # df_from_imdb = pd.read_csv(preprocessing_config['last_imdb_path'])

        # Сохранение сырого датасета
        df_from_imdb.to_csv(preprocessing_config['last_imdb_path'])

        # Предобработка
        df = transform_imdb_dataset(config_path, df_from_imdb)

        # Сохранение предобработанного датасета
        folder_path = preprocessing_config['df_dir_path']
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"file_{current_time}.csv"
        file_path = os.path.join(folder_path, file_name)
        df.to_csv(file_path)
        df.to_csv(preprocessing_config['last_train_path'])

        # Сохранение разных наборов уникальных значений
        data = df_from_imdb.replace('\\N', None)
        data = data.dropna()

        # Список уникальных жанров
        unique_genres = set()
        for lst in data['genres']:
            for genre in lst:
                unique_genres.add(genre)
        unique_genres = pd.DataFrame(unique_genres).rename(columns={0: 'unique_genres'})
        unique_genres.to_csv(unique_config['unique_genres_path'])

        # Список уникальных участников проекта
        unique_cast = set()
        for lst in data['cast']:
            for cast in lst:
                unique_cast.add(cast)
        unique_cast = pd.DataFrame(unique_cast).rename(columns={0: 'unique_cast'})
        unique_cast.to_csv(unique_config['unique_cast_path'])

        # Списки уникальных языков
        unique_OTLang = pd.DataFrame(data['originalTitle_language'].unique()).rename(columns={0: 'originalOTLang'})
        unique_OTLang.to_csv(unique_config['unique_ot_lang_path'])

        unique_PTLang = pd.DataFrame(data['primaryTitle_language'].unique()).rename(columns={0: 'primaryPTLang'})
        unique_PTLang.to_csv(unique_config['unique_pt_lang_path'])

    # Разбиение данных на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = split_train_test(
        x=df.drop(columns=[preprocessing_config['target_column']]),
        y=df[preprocessing_config['target_column']],
        shuffle=preprocessing_config['shuffle'],
        test_size=preprocessing_config['test_size'],
        random_state=preprocessing_config['random_state']
        # x_test_path=preprocessing_config['x_test_path']
    )

    # Поиск оптимальных гиперпараметров модели
    study = find_optimal_params(
        n_trials=train_config['n_trials'],
        x=x_train,
        y=y_train,
        shuffle=train_config['shuffle'],
        test_size=train_config['test_size'],
        random_state=train_config['random_state']
    )

    # Загрузка объекта study, хранящего в себе заранее подобранные гиперпараметры
    # study = joblib.load(os.path.join(train_config['study_path']))

    # Обучение с лучшими параметрами
    clf = train_model(
        x=x_train,
        y=y_train,
        x_test=x_test,
        y_test=y_test,
        study=study,
        metric_path=train_config['metrics_path'],
    )

    # Сохранение результатов
    joblib.dump(clf, os.path.join(train_config['model_path']))
    joblib.dump(study, os.path.join(train_config['study_path']))
