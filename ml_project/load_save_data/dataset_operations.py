"""Функция для работы с данными"""
import sys
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def read_data(path: str) -> pd.DataFrame:
    """Функция для загрузки датасета"""
    logger.info(f'Начинается загрузка файла {path}', )
    data = pd.read_csv(path)
    logger.info(f'Загрузка завершена {path}')
    return data


def split_train_test_data(data: pd.DataFrame, params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Функция для разбиение датасета на обучающую и тестовую выборки"""
    logger.info('Создание обучающей и тестовой выборки')
    train_data, test_data = train_test_split(
        data,
        test_size=params.test_size,
        random_state=params.random_state,
    )
    logger.info('Обучающая и тестовая выборка созданы')
    return train_data, test_data
