"""
Программа: Преобразование датасета imdb
Версия: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import yaml
import ast


def transform_imdb_dataset(config_path: str,
                           data: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка сведённого IMDB-датасета
    :param config_path: путь до конфигурационнго файла
    :param data: датасет
    :return: предобработанный датасет
    """
    # Загрузка конфигурационного файла
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']

    # Создание копии сырого датасета
    data_ = data.copy()

    # Замена строк вида \\N значением None
    data = data.replace('\\N', None)

    # Удаление объектов с пропущенными значениями в признаке 'год релиза'
    data = data.dropna(subset=[preprocessing_config['start_year_column']])

    # Заполнение пропусков в признаке 'экранное время в минутах' средним значением
    data[preprocessing_config['runtime_minutes_column']] = \
        data[preprocessing_config['runtime_minutes_column']].apply(lambda x: int(x) if not pd.isna(x) else x)
    data['runtimeMinutes'].fillna(value=data['runtimeMinutes'].mean(), inplace=True)

    # Заполнение пропусков в признаках 'список жанров' и 'список участников' пустыми списками
    data[preprocessing_config['cast_column']] = \
        data[preprocessing_config['cast_column']].apply(lambda x: [] if not isinstance(x, list) else x)
    data[preprocessing_config['genres_column']] = \
        data[preprocessing_config['genres_column']].apply(lambda x: [] if not isinstance(x, list) else x)

    # Поиск 100 самых популярных в датасете участников
    cast_count = {}
    for cast_list in data[preprocessing_config['cast_column']]:
        for cast in cast_list:
            if cast not in cast_count:
                cast_count[cast] = 1
            else:
                cast_count[cast] += 1
    popular_cast = sorted(cast_count.items(), key=lambda x: x[1], reverse=True)
    popular_cast = popular_cast[0:100]
    popular_cast = [x[0] for x in popular_cast]
    data = data.reset_index()

    # Бинаризация признака 'список участников' (выбираем 100 самых известных)
    mlb = MultiLabelBinarizer(classes=popular_cast)
    encoded_actors = mlb.fit_transform(data[preprocessing_config['cast_column']])
    sparse_matrix = csr_matrix(encoded_actors)
    df_encoded = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=mlb.classes_)
    data = pd.concat([data, df_encoded], axis=1)
    data.drop(preprocessing_config['cast_column'], axis=1, inplace=True)

    # Бинаризация признака 'список жанров'
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(data[preprocessing_config['genres_column']])
    sparse_matrix = csr_matrix(encoded_genres)
    df_encoded = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=mlb.classes_)
    data = pd.concat([data, df_encoded], axis=1)
    data.drop(preprocessing_config['genres_column'], axis=1, inplace=True)

    # Бинаризация категориальных признаков
    del data[preprocessing_config['index_column']]
    data = pd.get_dummies(data, columns=[preprocessing_config['title_type_column'],
                                         preprocessing_config['basics_primary_title_lang_column'],
                                         preprocessing_config['basics_original_title_lang_column']])

    # Приведение признака 'год релиза' к числовому типу
    data[preprocessing_config['start_year_column']] = \
        data[preprocessing_config['start_year_column']].apply(lambda x: int(x))

    # Загрузка датасета, содержащего информацию о списках фильмов, в которых участники принимали участие
    url = "https://datasets.imdbws.com/name.basics.tsv.gz"
    basics_names = pd.read_csv(url, sep="\t", compression="gzip")

    # Создание словаря для быстрого доступа к списку фильмов по id участника
    subset = data_[[preprocessing_config['movie_id_column'], preprocessing_config['target_column']]]
    subset = subset.set_index(preprocessing_config['movie_id_column'])
    d = subset.to_dict(orient=preprocessing_config['index_column'])

    # Замена в загруженном датасете списков id фильмов на списки рейтингов этих фильмов
    basics_names[preprocessing_config['basics_names_kft_column']] = \
        basics_names[preprocessing_config['basics_names_kft_column']].apply(lambda x: x.split(','))
    basics_names[preprocessing_config['basics_names_kft_column']] = \
        basics_names[preprocessing_config['basics_names_kft_column']].apply(
            lambda lst: [d[x]['averageRating'] if x in d else None for x in lst])

    # Формирование для каждого человека трёх характеристик качества - трёх статистик, взятых от уже числового списка
    basics_names['first_quality_of_person'] = basics_names[preprocessing_config['basics_names_kft_column']].apply(
        lambda x: np.mean(list(filter(None, x))))
    basics_names['second_quality_of_person'] = basics_names[preprocessing_config['basics_names_kft_column']].apply(
        lambda x: min(filter(None, x)) if len(list(filter(None, x))) > 0 else None)
    basics_names['third_quality_of_person'] = basics_names[preprocessing_config['basics_names_kft_column']].apply(
        lambda x: max(filter(None, x)) if len(list(filter(None, x))) > 0 else None)

    # Сохранение полученного датасета для дальнейшего использования в методе предсказания
    basics_names.to_csv(preprocessing_config['basics_names_fe_path'])

    # Присоединение полученных характеристик к основному датасету
    subset_data = data_[[preprocessing_config['movie_id_column'], preprocessing_config['cast_column']]]
    subset_data[preprocessing_config['cast_column']] = \
        subset_data[preprocessing_config['cast_column']].apply(lambda x: [] if not isinstance(x, list) else x)

    subset = basics_names[[preprocessing_config['person_id_column'],
                           'first_quality_of_person',
                           'second_quality_of_person',
                           'third_quality_of_person']]
    subset = subset.set_index(preprocessing_config['person_id_column'])
    d = subset.to_dict(orient=preprocessing_config['index_column'])
    subset_data['cast_first_quality'] = subset_data[preprocessing_config['cast_column']].apply(
        lambda lst: [d[x]['first_quality_of_person'] if (not pd.isna(x) and (x in d)) else None for x in lst])
    subset_data['cast_second_quality'] = subset_data['cast'].apply(
        lambda lst: [d[x]['second_quality_of_person'] if (not pd.isna(x) and (x in d)) else None for x in lst])
    subset_data['cast_third_quality'] = subset_data['cast'].apply(
        lambda lst: [d[x]['third_quality_of_person'] if (not pd.isna(x) and (x in d)) else None for x in lst])

    subset_data['fq_mean'] = subset_data['cast_first_quality'].apply(lambda x: np.mean(list(filter(None, x))))
    subset_data['fq_min'] = subset_data['cast_first_quality'].apply(
        lambda x: min(filter(None, x)) if len(list(filter(None, x))) > 0 else None)
    subset_data['fq_max'] = subset_data['cast_first_quality'].apply(
        lambda x: max(filter(None, x)) if len(list(filter(None, x))) > 0 else None)

    subset_data['sq_mean'] = subset_data['cast_second_quality'].apply(lambda x: np.mean(list(filter(None, x))))
    subset_data['sq_min'] = subset_data['cast_second_quality'].apply(
        lambda x: min(filter(None, x)) if len(list(filter(None, x))) > 0 else None)
    subset_data['sq_max'] = subset_data['cast_second_quality'].apply(
        lambda x: max(filter(None, x)) if len(list(filter(None, x))) > 0 else None)

    subset_data['tq_mean'] = subset_data['cast_third_quality'].apply(lambda x: np.mean(list(filter(None, x))))
    subset_data['tq_min'] = subset_data['cast_third_quality'].apply(
        lambda x: min(filter(None, x)) if len(list(filter(None, x))) > 0 else None)
    subset_data['tq_max'] = subset_data['cast_third_quality'].apply(
        lambda x: max(filter(None, x)) if len(list(filter(None, x))) > 0 else None)

    subset_data.drop(columns=[preprocessing_config['cast_column'],
                              'cast_first_quality',
                              'cast_second_quality',
                              'cast_third_quality'], inplace=True)

    data = pd.merge(data, subset_data, on=preprocessing_config['movie_id_column'])

    # Заполнение пропусков средним в добавленных столбцах
    data['fq_mean'] = data['fq_mean'].fillna(data['fq_mean'].mean())
    data['fq_min'] = data['fq_min'].fillna(data['fq_min'].mean())
    data['fq_max'] = data['fq_max'].fillna(data['fq_max'].mean())

    data['sq_mean'] = data['sq_mean'].fillna(data['sq_mean'].mean())
    data['sq_min'] = data['sq_min'].fillna(data['sq_min'].mean())
    data['sq_max'] = data['sq_max'].fillna(data['sq_max'].mean())

    data['tq_mean'] = data['tq_mean'].fillna(data['tq_mean'].mean())
    data['tq_min'] = data['tq_min'].fillna(data['tq_min'].mean())
    data['tq_max'] = data['tq_max'].fillna(data['tq_max'].mean())
    del data[preprocessing_config['movie_id_column']]

    # Удаление пропусков в бинарных признаках, если они вдруг есть
    data.dropna(inplace=True)

    # Приведение признака 'наличие ограничения по возрасту' к числовому типу
    data[preprocessing_config['is_adult_column']] = \
        data[preprocessing_config['is_adult_column']].apply(lambda x: int(x))

    return data


def transform_imdb_object(data: pd.DataFrame,
                          config_path: str,
                          num_of_new_people: int=0) -> pd.DataFrame:
    """
    Предобразование одного объекта (предполагается, что он поступает на вход модели)
    :param config_path: путь до конфигурационного файла
    :param data: датасет
    :param num_of_new_people: число участников проекта, которых нет в БД
    :return: предобработанный датасет
    """
    # Загрузка конфигурационного файла
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    unique_config = config['unique_data']

    # Загрузка последнего сырого датасета
    last_imdb = pd.read_csv(preprocessing_config['last_imdb_path'])

    # Удаление пропусков в сырого датасете
    last_imdb.dropna(inplace=True)

    # Преобразование признака 'список участников к списковому типу
    last_imdb[preprocessing_config['cast_column']] = \
        last_imdb[preprocessing_config['cast_column']].apply(lambda x: ast.literal_eval(x))

    ''' Замена признака 'список участников' на 100 бинарных признаков, каждый из которых представляет собой
     наличие одного из 100 самых популярных участников в последнем сохранённом сыром датасете'''
    cast_count = {}
    for cast_list in last_imdb[preprocessing_config['cast_column']]:
        for cast in cast_list:
            if cast not in cast_count:
                cast_count[cast] = 1
            else:
                cast_count[cast] += 1

    popular_cast = sorted(cast_count.items(), key=lambda x: x[1], reverse=True)
    popular_cast = popular_cast[0:100]
    popular_cast = [x[0] for x in popular_cast]
    for x in popular_cast:
        data[x] = 1 if x in data[preprocessing_config['cast_column']] else 0

    # Преобразование признака 'список жанров'
    unique_genres = pd.read_csv(unique_config['unique_genres_path'])
    for x in list(unique_genres['unique_genres']):
        data[x] = 1 if x in data[preprocessing_config['genres_column']] else 0
    del data[preprocessing_config['genres_column']]

    # Преобразование признака 'язык оригинального названия'
    unique_ot_lang = pd.read_csv(unique_config['unique_ot_lang_path'])
    for x in list(unique_ot_lang['originalOTLang']):
        data['originalTitle_language_' + x] = \
            1 if x in data[preprocessing_config['basics_original_title_lang_column']] else 0
    del data[preprocessing_config['basics_original_title_lang_column']]

    # Преобразование признака 'язык популярного названия'
    unique_pt_lang = pd.read_csv(unique_config['unique_pt_lang_path'])
    for x in list(unique_pt_lang['primaryPTLang']):
        data['primaryTitle_language_' + x] = \
            1 if x in data[preprocessing_config['basics_primary_title_lang_column']] else 0
    del data[preprocessing_config['basics_primary_title_lang_column']]

    # Преобразование признака 'год релиза'
    data[preprocessing_config['start_year_column']] = \
        data[preprocessing_config['start_year_column']].apply(lambda x: int(x) if not pd.isna(x) else x)

    # Преобразовани признака 'тип заголовка'
    data['titleType_tvMovie'] = 1 if 0 in data[preprocessing_config['title_type_column']] else 0
    data['titleType_movie'] = 1 if 1 in data[preprocessing_config['title_type_column']] else 0
    del data['titleType']

    # Получение средних значений сгенерированных признаков в обучающем датасете
    last_data = pd.read_csv(preprocessing_config['last_train_path'])
    fq_mean = last_data['fq_mean'].mean()
    sq_mean = last_data['sq_mean'].mean()
    tq_mean = last_data['tq_mean'].mean()
    fq_min = last_data['fq_min'].mean()
    sq_min = last_data['sq_min'].mean()
    tq_min = last_data['tq_min'].mean()
    fq_max = last_data['fq_max'].mean()
    sq_max = last_data['sq_max'].mean()
    tq_max = last_data['tq_max'].mean()

    if len(data['cast']) == 0:
        data['fq_mean'] = fq_mean
        data['sq_mean'] = sq_mean
        data['tq_mean'] = tq_mean
        data['fq_min'] = fq_min
        data['sq_min'] = sq_min
        data['tq_min'] = tq_min
        data['fq_max'] = fq_max
        data['sq_max'] = sq_max
        data['tq_max'] = tq_max
    else:
        # Загрузка датасета, содержащего информацию о списках фильмов, в которых участники принимали участие
        basics_names_fe = pd.read_csv(preprocessing_config['basics_names_fe_path'])

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                   ['first_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                      fq_mean] * num_of_new_people
        data['fq_mean'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                   ['second_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                      sq_mean] * num_of_new_people
        data['sq_mean'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                   ['third_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                      tq_mean] * num_of_new_people
        data['tq_mean'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                 ['first_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                    fq_min] * num_of_new_people
        data['fq_min'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                 ['second_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                    sq_min] * num_of_new_people
        data['sq_min'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                 ['third_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                    tq_min] * num_of_new_people
        data['tq_min'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                 ['first_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                    fq_max] * num_of_new_people
        data['fq_max'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                 ['second_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                    sq_max] * num_of_new_people
        data['sq_max'] = np.mean(lst)

        lst = [basics_names_fe[basics_names_fe['nconst'] == x]
                                 ['third_quality_of_person'] for x in data['cast'].iloc[0]] + [
                                    tq_max] * num_of_new_people
        data['tq_max'] = np.mean(lst)

    del data[preprocessing_config['cast_column']]
    data['isAdult'] = data['isAdult'].apply(lambda x: int(x))

    return data
