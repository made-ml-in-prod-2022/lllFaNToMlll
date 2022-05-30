"""
Файл с функциями для предобработки датасета
"""
import sys
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def one_hot_encoding() -> Pipeline:
    """Фукнция для создания OneHotEncoding"""
    return Pipeline([('OHE', OneHotEncoder())])


def extract_target(data: pd.DataFrame, params) -> pd.Series:
    """Фукнция извлечения таргета"""
    return data[params.target]


def transform_dataset(transformer: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    """Фукнция кодирования данных с помощью трансформера"""
    return transformer.transform(data)


def drop_target(data: pd.DataFrame, params) -> pd.DataFrame:
    """Функция удаления таргета из датасета"""
    return data.drop(columns=[params.target])


def build_feature_transformer(params) -> ColumnTransformer:
    """Фукнция создания трансформера"""
    return ColumnTransformer([
        (
            'one_hot_encoding',
            one_hot_encoding(),
            params.one_hot_encoding_features
        )
    ])
