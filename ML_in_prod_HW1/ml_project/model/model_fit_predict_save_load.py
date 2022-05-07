import pickle
import sys
import logging
from typing import Dict, NoReturn
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_model(features: pd.DataFrame, target: pd.Series, train_params) -> XGBClassifier:
    logger.info(f'Обучение модели {train_params.model_type}')
    if train_params.model_type == 'XGBClassifier':
        model = XGBClassifier(
            random_state=train_params.random_state,
            learning_rate=train_params.learning_rate,
            max_depth=train_params.max_depth,
            n_estimators=train_params.n_estimators,
            eval_metric=train_params.eval_metric,
            use_label_encoder=train_params.use_label_encoder
        )
    elif train_params.model_type == 'LogisticRegression':
        model = LogisticRegression(
            random_state=train_params.random_state,
            penalty=train_params.penalty,
            max_iter=train_params.max_iter,
            solver=train_params.solver
        )
    else:
        logger.exception('Недопустимая модель')
        raise NotImplementedError()
    model.fit(features, target)
    logger.info('Обучение завершено')
    return model


def predict_model(model: XGBClassifier, feature: pd.DataFrame) -> np.ndarray:
    logger.info('Предсказание модели')
    predict = model.predict(feature)
    logger.info('Модель завершила предсказания')
    return predict


def calculate_metrics(predict: np.ndarray, target: pd.Series,
                      path: str = "models/XGBClassifier_metrics_train.json") -> Dict[str, float]:
    logger.info('Расчет метрик')
    metrics = {
        "Roc|Auc": roc_auc_score(target, predict),
        "Accuracy": accuracy_score(target, predict),
        "F1": f1_score(target, predict)
    }
    with open(path, 'w') as fp:
        json.dump(metrics, fp)

    return metrics


def save_model(model: XGBClassifier, path: str = "models/XGBClassifier.pkl") -> NoReturn:
    with open(path, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Модель сохранена')


def load_model(path: str = "models/XGBClassifier.pkl"):
    logger.info('Загрузка модели')
    with open(path, 'rb') as model:
        return pickle.load(model)


def save_transformer(transformer, path: str = "models/XGBClassifier_transformer.pkl"):
    with open(path, 'wb') as file:
        pickle.dump(transformer, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Transformer сохранен')


def load_transformer(path: str):
    logger.info('Загрузка transformer')
    with open(path, 'rb') as transformer:
        return pickle.load(transformer)
