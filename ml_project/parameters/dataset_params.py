from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataSetParams:
    one_hot_encoding_features: List[str]
    num_features: List[str]
    target: Optional[str]


@dataclass
class SplittingParams:
    test_size: float = field(default=0.20)
    random_state: int = field(default=42)


@dataclass
class TransformerParams:
    use_custom_transformer: bool
