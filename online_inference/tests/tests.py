from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from online_inference.app import app


def test_start():
    """Проверка функции get"""
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200


def test_health():
    """Проверка функции health"""
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json() is True


def test_predict():
    """Проверка работы функции предсказания"""
    real_response = [
        [{'id': 0, 'target': 0}],
        [{'id': 0, 'target': 1}],
    ]

    with TestClient(app) as client:
        data = pd.read_csv('data/test_data.csv')
        request_features = list(data.columns)
        for i, _ in enumerate(data.shape):
            request_data = [
                x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
            ]

            response = client.get(
                '/predict',
                json={
                    'data': [request_data],
                    'features': request_features,
                },
            )
            assert response.status_code == 200
            assert response.json() == real_response[i]


def test_bad_command():
    """Проверка на корректную обработку неправильной команды"""
    with TestClient(app) as client:
        response = client.get('/badcommand')
        assert response.status_code == 404
