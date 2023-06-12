"""
Программа: Получение данных заданному по пути и напрямую от пользователя
Версия: 1.0
"""

from io import BytesIO
import io
from typing import Dict, Tuple
import streamlit as st
import pandas as pd


def get_data(df_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param df_path: путь
    :return: датасет
    """
    return pd.read_parquet(df_path)


def load_data(
        data: str,
        type_data: str,
) -> Dict[str, Tuple[str, BytesIO, str]]:
    """
    Получение данных и преобразование в тип BytesIO
    :param data: путь
    :param type_data:тип датасета (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_parquet(data)
    st.write("Dataset_load")
    st.write(dataset.head())

    dataset_bytes_obj = io.BytesIO()
    dataset.to_parquet(dataset_bytes_obj, index=False)
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.parquet", dataset_bytes_obj, "multipart/form-data")
    }
    return files