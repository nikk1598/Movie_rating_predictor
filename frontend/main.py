"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import streamlit as st
import yaml
import sys
import os
import pandas as pd

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# project_path = os.path.join(project_root, 'movie_rating_predictor')
# sys.path.append(project_path)

from src.train.train import start_training
from src.evaluate.evaluate_input import evaluate_input

st.set_option('depr''ecation.showPyplotGlobalUse', False)
CONFIG_PATH = 'config/params.yml'


def main_page():
    """
    Страница с описанием проекта
    :return:
    """
    st.image(
        "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmY3ZjgxNGRlMWQ1MDY3MzA0M2Q0ZWIzZGZiZWZhNDIyZmJhNGE5YiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/N25oDzBxS8KbRf4iIG/giphy.gif",
        width=900
    )

    st.markdown("# Прогнозирование рейтингов фильмов")
    st.write(
        """
       С помощью данного приложения вы можете внести информацию о новом фильме, и получить прогноз его IMDB-рейтинга\n
       """
    )
    st.markdown("# Описание меню")
    st.write("Training model - обучение ML-модели LightGBM")
    st.write("Prediction - внесение данных и получение предсказания ")


def training():
    """
    Обучение модели
    """
    with open(CONFIG_PATH, encoding='utf8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    st.markdown("# Обучение на данных IMDB")
    st.write("Программа загрузит данные из IMDB, сформирует и сохранит датасет для обучения, "
             "обучит и сохранит модель. Скрипт требует 32 ГБ оперативной памяти. При нехватке ресурсов воспользуйтесь"
             " инструкцией https://colab.research.google.com/drive/1I3cd0ZG8xQjBMwiUiJn2m5Abvo979jCf?usp=sharing, "
             "а затем обучите модель на готовом датасате.\n")

    if st.button("Start training"):
        start_training(
            config=config,
            df=pd.DataFrame(),
            empty_df=True
        )
    st.markdown("# Обучение на готовом датасете")
    st.write("Загрузите ранее сгенерированный датасет.", )

    upload_file = st.file_uploader(
        "", type=['csv'], accept_multiple_files=False
    )
    if upload_file:
        df = pd.read_csv(upload_file)
        del df['Unnamed: 0']
        if st.button("Start training "):
            start_training(
                config=config,
                df=df,
                empty_df=False
            )


def prediction():
    """
    Получение предсказаний модели по введённым данным
    """
    with open(CONFIG_PATH, encoding='utf8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    st.markdown("# Введите данные о фильме")

    evaluate_input(config['unique_data']['unique_genres_path'],
                   config['unique_data']['unique_pt_lang_path'],
                   config['unique_data']['unique_ot_lang_path'],
                   config['unique_data']['unique_cast_path']
                   )


if __name__ == "__main__":
    page_names = {
        'Описание проекта': main_page,
        'Training model': training,
        'Prediction': prediction,
    }
    selected_page = st.sidebar.selectbox('Выберите пункт', page_names.keys())
    page_names[selected_page]()
