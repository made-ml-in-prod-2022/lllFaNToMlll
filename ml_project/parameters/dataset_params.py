"""Файл с датаклассами параметров для работы с датасетом"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataSetParams:
    """Датакласс параметров для датасета"""
    one_hot_encoding_features: List[str]
    num_features: List[str]
    target: Optional[str]


@dataclass
class SplittingParams:
    """Датакласс параметров для разбиения датасета"""
    test_size: float = field(default=0.20)
    random_state: int = field(default=42)


@dataclass
class TransformerParams:
    """Датакласс параметров для применения
    кастомного трансформера"""
    use_custom_transformer: bool
