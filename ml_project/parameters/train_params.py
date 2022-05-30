"""Файл с датаклассом параметров для этапа обучения"""
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .dataset_params import TransformerParams, SplittingParams, DataSetParams
from .model_params import ModelParams


@dataclass
class TrainingParams:
    """Датакласс параметров для этапа предсказания"""
    input_data_path: str
    metric_path: str
    save_model: str
    save_transformer: str
    model_params: ModelParams
    custom_transformer_params: TransformerParams
    feature_params: DataSetParams
    splitting_params: SplittingParams


TrainingParamsSchema = class_schema(TrainingParams)


def read_training_params(path: str):
    """Функция для чтения параметров"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
