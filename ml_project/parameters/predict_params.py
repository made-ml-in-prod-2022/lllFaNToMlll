"""Файл с датаклассом параметров для этапа предсказания"""
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictParams:
    """Датакласс параметров для этапа предсказания"""
    input_data_path: str
    model_path: str
    predict_path: str
    transformer_path: str
    target_in_dataset: bool
    target: str


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(path: str):
    """Функция для чтения параметров"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
