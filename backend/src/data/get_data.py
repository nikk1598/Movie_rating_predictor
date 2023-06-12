"""
Программа: Методы получения данных
Версия: 1.0
"""

import langid
import yaml
from typing import Text, Union, BinaryIO
import pandas as pd


def get_dataset(dataset_path: Union[Text, BinaryIO]) -> pd.DataFrame:
    """
    Получение данных в формате parquet по заданному пути
    :param dataset_path: путь до датасета
    :return: датасет
    """
    return pd.read_parquet(dataset_path)


def get_imdb_dataset(config_path: str) -> pd.DataFrame:
    """
    Получение сведённого датасета IMDB с сайта
    :param config_path: путь до конфигурационного файла
    :return: датасет
    """
    # Загрузка конфигурационного файла
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']

    # Загрузка датасета с информацией о рейтингах
    url = preprocessing_config['ratings_dataset_url']
    ratings = pd.read_csv(url, sep="\t", compression="gzip")

    # Удаление ненужного столбца с числом голосов
    del ratings[preprocessing_config['ratings_num_votes_column']]

    # Загрузка датасета с информацией о фильмах
    url = preprocessing_config['basics_dataset_url']
    basics = pd.read_csv(url, sep="\t", compression="gzip")

    # Отбрасывание заголовков, которые не являются фильмами
    basics = basics[(basics.titleType == preprocessing_config['basics_movie_type']) |
                    (basics.titleType == preprocessing_config['basics_tv_movie_type'])]

    # Извлечение языковых признаков из столбцов primaryTitle и originalTitle
    basics[preprocessing_config['basics_primary_title_lang_column']] =\
        basics[preprocessing_config['basics_primary_title_column']].astype(str).apply(lambda x: langid.classify(x)[0])
    basics['originalTitle_language'] = basics['originalTitle'].astype(str).apply(lambda x: langid.classify(x)[0])

    # Удаление ненужных признаков
    del basics[preprocessing_config['basics_primary_title_column']]
    del basics[preprocessing_config['basics_original_title_column']]
    del basics[preprocessing_config['basics_end_year_column']]

    # Соединение двух датасетов
    data = pd.merge(basics, ratings)

    # Загрузка датасета с информацией об участниках проектов
    url = preprocessing_config['principals_dataset_url']
    principals = pd.read_csv(url, sep="\t", compression="gzip")

    # Добавление к основному датасету информации о списках участников
    principals_grouped = pd.DataFrame(principals.groupby(preprocessing_config['movie_id_column'])
                                      [preprocessing_config['person_id_column']].agg(list))
    principals_cast_grouped = principals_grouped.rename(columns={preprocessing_config['person_id_column']: 'cast'})
    data_upd = pd.merge(data, principals_cast_grouped, on=preprocessing_config['movie_id_column'], how='left')

    # Приведение признака genres к списковому типу
    data_upd[preprocessing_config['genres_column']] = \
        data_upd[preprocessing_config['genres_column']].apply(lambda x: x.split(',') if not pd.isna(x) else x)

    return data_upd
