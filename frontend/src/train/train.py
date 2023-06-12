"""
Программа: Обучение модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import json
import sys
import os
import joblib
import pandas as pd
import streamlit as st
from optuna.visualization import plot_optimization_history
from lightgbm import plot_importance
from PIL import Image

#current_dir = os.path.dirname(os.path.abspath(__file__))
#project_root = os.path.dirname(os.path.dirname(current_dir))
#project_path = os.path.join(project_root, 'movie_rating_predictor')
#sys.path.append(project_path)

from backend import main_


def start_training(config: dict,
                   df: pd.DataFrame,
                   empty_df=True) -> None:
    """
    Обучение модели с выводом результатов
    :param df: датасет
    :param config: конфигурационный файл
    :param empty_df: параметр, указывающий на то, был ли передан датасет для обучения
    :return: None
    """
    # Открываем файл с метриками, если он есть, и сохраняем их в переменную
    if os.path.exists(config['training']['metrics_path']):
        with open(config['training']['metrics_path']) as json_file:
            old_metrics = json.load(json_file)
    # Если файла нет, то зануляем все метрики
    else:
        old_metrics = {'mae': 0,
                       'mse': 0,
                       'r2_adjusted': 0,
                       'wape': 0}
    ''' 
    Показываем колесо и отправляем запрос (в main сюда будет 
    передан адрес, по которому пользователь может обучить модель,
    соответственно медод из модуля main бэкэнда начнёт выполнять соответствующий код)
    '''
    with st.spinner('Идёт обучение модели...'):
        '''
        В output запишется то, что возвращает сервер в ответ на запрос
        (как мы помним, для запроса на обучение это словарь с метриками плюс
        в old_metrics у нас уже сохранены предыдущие метрики, либо нули)
        '''
        new_metrics = main_.training(df, empty_df=empty_df)
    st.success('Успех!')

    '''
    Представляем наши метрики в формате специальной таблицы streamlit,
    для каждого столбца передаём название метрики, новую подсчитанную метрику,
    а также разницу между старой и новой метрикой
    '''

    mae, mse, r2_adjusted, wape = st.columns(4)
    mae.metric(
        'mae',
        new_metrics['mae'],
        f"{new_metrics['mae'] - old_metrics['mae']:.3f}"
    )
    mse.metric(
        'mse',
        new_metrics['mse'],
        f"{new_metrics['mse'] - old_metrics['mse']:.3f}"
    )
    r2_adjusted.metric(
        'r2_adjusted',
        new_metrics['r2_adjusted'],
        f"{new_metrics['r2_adjusted'] - old_metrics['r2_adjusted']:.3f}"
    )
    wape.metric(
        'wape, %',
        new_metrics['wape']*100,
        f"{new_metrics['wape']*100 - old_metrics['wape']*100:.3f}"
    )

    # Показываем графики с результатами обучения
    study = joblib.load(os.path.join(config['training']['study_path']))
    clf = joblib.load(os.path.join(config['training']['model_path']))

    fig_history = plot_optimization_history(study)
    st.plotly_chart(fig_history, use_container_width=True)

    feat_imp = plot_importance(clf, max_num_features=10, figsize=(20, 10))
    feat_imp.figure.savefig(config['report']['feat_imp_path'])

    image = Image.open(config['report']['feat_imp_path'])
    st.image(image, caption='Features')
