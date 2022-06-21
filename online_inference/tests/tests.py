import unittest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from online_inference.app import app


class MyTestCase(unittest.TestCase):
    def test_start(self):
        """Проверка функции get"""
        with TestClient(app) as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)

    def test_health(self):
        """Проверка функции health"""
        with TestClient(app) as client:
            response = client.get('/health')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json())

    def test_predict(self):
        """Проверка работы функции предсказания"""
        real_response = [
            [{'id': 0, 'target': 0}],
            [{'id': 0, 'target': 1}],
        ]

        with TestClient(app) as client:
            data = pd.read_csv('../data/test_data.csv')
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
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), real_response[i])

    def test_bad_command(self):
        """Проверка на корректную обработку неправильной команды"""
        with TestClient(app) as client:
            response = client.get('/badcommand')
            self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
