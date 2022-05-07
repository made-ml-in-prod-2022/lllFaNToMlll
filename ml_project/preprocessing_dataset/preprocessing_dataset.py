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
    return Pipeline([('OHE', OneHotEncoder())])


def extract_target(data: pd.DataFrame, params) -> pd.Series:
    return data[params.target]


def transform_dataset(transformer: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(data)


def drop_target(data: pd.DataFrame, params) -> pd.DataFrame:
    return data.drop(columns=[params.target])


def build_feature_transformer(params) -> ColumnTransformer:
    return ColumnTransformer([
        (
            'one_hot_encoding',
            one_hot_encoding(),
            params.one_hot_encoding_features
        )
    ])
