# Домашнее задание 1 по курсу Машинное обучение в продакшене
Установка: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Обучение XGBClassifier:
~~~
python ml_project/model_train.py configs/XGBClassifier_train_config.yaml
~~~
Предсказание XGBClassifier:
~~~
python ml_project/model_predict.py configs/XGBClassifier_predict_config.yaml
~~~
Обучение LogisticRegression:
~~~
python ml_project/model_train.py configs/LogisticRegression_train_config.yaml
~~~
Предсказание LogisticRegression:
~~~
python ml_project/model_predict.py configs/LogisticRegression_predict_config.yaml
~~~
Организация проекта
------------
	├── configs                 <- Конфиги с параметрами для запуска моделей.
    │
    ├── data
    │   ├── processed.csv       <- Предобработанные данные.
    │   └── raw.csv             <- Реальные данные.
    │
	├── ml_project              <- Код для запуска пайплана.
    │   │
    │   ├── load_save_data      <- Загрузка и сохранение датасета.
    │   │
	│	├── model 	            <- Обучение модели, а также ее сохранение и загрузка.
	│	│
    │   ├── parameters          <- dataclasses для работы.
    │   │
    │   ├── preprocessing_dataset  <- Предобработка датасета.
    │   │
    │   ├── model_predict.py    <- Файл, запускающий модель в режиме предсказания.
    │   │
    │   ├── model_train.py   <- Файл, запускающий модель в режиме обучения.
	│
    ├── models                  <- Сохраненные модели, трансформеры, метрики, предсказания.
    │
    ├── notebooks               <- EDA и базовый пайплан в Jupyter notebook.
    │
	├── tests                   <- Тесты.
	│
	├── README.md               <- Инструкция к проекту.
	│
    ├── requirements.txt        <- Необходимые библиотеки для работы проекта.
    │
    ├── setup.py                <- Возможность установки проекта через менеджер pip.

--------