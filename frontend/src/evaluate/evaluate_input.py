"""
Программа: Отрисовка слайдеров и кнопок для ввода пользовательских данных
Версия: 1.0
"""

import pandas as pd
import streamlit as st
import sys
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# project_path = os.path.join(project_root, 'movie_rating_predictor')
# sys.path.append(project_path)

from backend.src.data.transform_data import transform_imdb_object
from backend import main_

CONFIG_PATH = 'config/params.yml'


def evaluate_input(unique_genres_path: str,
                   unique_pt_lang_path: str,
                   unique_ot_lang_path: str,
                   unique_cast_path: str):
    """
    Получение предсказаний и вывод результата
    :param unique_genres_path: путь к списку уникальных жанров
    :param unique_pt_lang_path: путь к списку уникальных языков оригинала
    :param unique_ot_lang_path: путь к списку уникальных языков для названия, использующихся в маркетинге
    :param unique_cast_path: путь к списку уникальных участников проектов
    :return:
    """

    title_type = st.selectbox(
        "Показ в кинотеатрах", ['Да', 'Нет']
    )
    title_type = 1 if (title_type == 'Да') else 0

    unique_genres_df = pd.read_csv(unique_genres_path)
    unique_genres = unique_genres_df['unique_genres']
    genres = st.multiselect(
        "Список жанров",
        unique_genres
    )

    is_adult = st.selectbox(
        "Ограничение по возрасту 18+", ['Да', 'Нет']
    )
    is_adult = 1 if (is_adult == 'Да') else 0

    unique_cast_df = pd.read_csv(unique_cast_path)
    unique_cast = unique_cast_df['unique_cast'][0:2000]
    cast = st.multiselect(
        "Список участников команды проекта",
        unique_cast
    )
    num_of_new_people = st.number_input("Количество участников, которых нет в БД", value=0, step=1)

    runtime_minutes = st.number_input("Экранное время в минутах", value=90, step=10)

    start_year = st.number_input("Год выпуска", value=2023, step=1)

    unique_pt_lang_df = pd.read_csv(unique_pt_lang_path)
    unique_pt_lang = unique_pt_lang_df['primaryPTLang']
    pt_language = st.selectbox(
        "Язык оригинала", unique_pt_lang
    )

    unique_ot_lang_df = pd.read_csv(unique_ot_lang_path)
    unique_ot_lang = unique_ot_lang_df['originalOTLang']
    ot_language = st.selectbox(
        "Язык названия, использующийся в маркетинге", unique_ot_lang
    )

    st.write(
        f"""### Ваши данные:\n
    1) Показ в кинотеатрах: {title_type}
    2) Список жанров, сопоставленных фильму: {genres}
    3) Ограничение по возрасту: {is_adult}
    4) Список участников команды проекта: {cast}
    5) Количество участников которых нет в БД: {num_of_new_people}
    6) Экранное время в минутах: {runtime_minutes}
    7) Год выпуска: {start_year}
    8) Язык оригинала: {pt_language}
    9) Язык названия, использующийся в маркетинге: {ot_language}
    """)

    movie = pd.DataFrame(columns=['titleType',
                         'isAdult',
                         'startYear',
                         'runtimeMinutes',
                         'genres',
                         'primaryTitle_language',
                         'originalTitle_language',
                         'cast'])

    movie.loc[len(movie.index)] = [title_type,
                                   is_adult,
                                   start_year,
                                   runtime_minutes,
                                   genres,
                                   pt_language,
                                   ot_language,
                                   cast]

    button_ok = st.button("Predict")
    if button_ok:
        if len(cast) == 0:
            st.write('Список участников не может быть пустым')
        elif (num_of_new_people < 0) or (runtime_minutes < 0) or (start_year < 0):
            st.write('Значения не могут быть отрицательными')
        else:
            with st.spinner('Модель делает предсказание...'):
                movie_transformed = transform_imdb_object(movie, CONFIG_PATH)
                predict = main_.prediction(movie_transformed)
                st.success('Успех!')
                st.write(predict)


