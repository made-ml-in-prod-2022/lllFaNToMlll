"""Файл с датаклассом параметров модели"""
from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    """Датакласс параметров модели"""
    # Общие параметры
    model_type: str = field(default="XGBClassifier")
    random_state: int = field(default=42)
    # Параметры XGBClassifier
    learning_rate: int = field(default=0.0001)
    max_depth: int = field(default=5)
    n_estimators: int = field(default=1000)
    eval_metric: str = field(default="logloss")
    use_label_encoder: bool = field(default=False)
    # Параметры LogReg
    penalty: str = field(default='l1')
    max_iter: int = field(default=5000)
    solver: str = field(default="liblinear")
