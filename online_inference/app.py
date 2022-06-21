import pickle
import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
from sklearn.pipeline import Pipeline
from source.response import PredictResponse, InputDataRequest


def transform_dataset(transformer, data):
    """Фукнция кодирования данных с помощью трансформера"""
    return transformer.transform(data)


def load_transformer(path: str):
    """Функция для загрузки трансформера"""
    logger.info('Загрузка transformer')
    with open(path, 'rb') as transformer:
        return pickle.load(transformer)

MODEL = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] => %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('ml_in_prod_hw2')

app = FastAPI()


@app.get('/')
async def start():
    """Начало работы"""
    return 'FastAPI запущен'


@app.get('/health')
def health() -> bool:
    """Функция для получения статуса модели"""
    logger.info('Получение статуса модели')
    if MODEL:
        return True
    else:
        return False


@app.on_event('startup')
async def load_model():
    """Загрузка модели"""
    global MODEL

    model_path = os.getenv(
      'PATH_TO_MODEL',
       default='models/XGBClassifier.pkl'
    )

    logger.info(f'Начало загрузки {model_path} модели')

    with open(model_path, 'rb') as model:
        MODEL = pickle.load(model)

    logger.info(f'Модель {model_path} загружена')


def make_predict(data: List, features: List[str], model: Pipeline)\
        -> List[PredictResponse]:
    """Функция для получения результатов предсказания"""
    logger.info(f'Начало предсказания модели')
    data = pd.DataFrame(data, columns=features)
    transformer = load_transformer("models/XGBClassifier_transformer.pkl")
    feature = transform_dataset(transformer, data)
    n_row = [i for i, _ in enumerate(feature)]
    predictions = model.predict(feature)
    logger.info(f'Модель завершира предсказания')
    return [
        PredictResponse(id=index, target=target) for index, target in zip(n_row, predictions)
    ]


@app.get('/predict')
def predict(request: InputDataRequest) -> List[PredictResponse]:
    """Вызов модели для предсказания"""
    if health():
        logger.info(f'Вызов функции предсказания')
        return make_predict(request.data, request.features, MODEL)

    else:
        logger.error('Модель не работает')
        raise HTTPException(
            status_code=404,
            detail='Модель не найдена'
        )


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=os.getenv('PORT', 80))
