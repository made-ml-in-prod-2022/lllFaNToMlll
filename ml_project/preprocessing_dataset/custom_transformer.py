import logging
from typing import NoReturn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features) -> NoReturn:
        self.scaler = MinMaxScaler()
        self.num_features = features.num_features

    def fit(self, data: pd.DataFrame):
        self.scaler.fit(data[self.num_features])
        return self

    def transform(self, data: pd.DataFrame):
        return self.scaler.transform(data[self.num_features])
