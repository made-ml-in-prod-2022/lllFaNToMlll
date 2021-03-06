# Домашнее задание 3 по курсу Машинное обучение в продакшене
## Airflow
Для windows:
- Запускаем Docker Desktop
- В терминале, находясь в папке airflow_ml_dags, пишем
~~~
docker-compose up --build
~~~
- Ждем формирования контейнера, а также создания локального Airflow
- Переходим в браузере на http://localhost:8080/
- Запускаем по очереди DAG'и
- В терминале, находясь в папке airflow_ml_dags, для завершения работы пишем
~~~
docker-compose down
~~~
Организация проекта
------------
    ├── dags                                <- Папка с DAG'ами.
    │   │
    │   ├── 01_generate_data                <- DAG для загрузки датасет.
    │   │
    │   ├── 02_train_model                  <- DAG для подготовки датасета, предобработки, обучения и подсчета метрик.
    │   │
    │   ├── 03_make_predict                 <- DAG для предсказания.
    │
    ├── data                                <- Папка для артефактов.
    │
    ├── images                              <- Папка с реализацией всех этапов, а также с докерфайлами.
    │   │
    │   ├── airflow-docker                  <- Докерфайл для airflow.
    │   │
    │   ├── airflow-generate-data           <- Докерфайл для airflow и скрипт для загрузки датасета.
    │   │
    │   ├── airflow-ml-base                 <- Докерфайл для airflow и requirements.
    │   │
    │   ├── airflow-predict                 <- Докерфайл для airflow и скрипт для предсказания.
    │   │
    │   ├── airflow-preprocessing           <- Докерфайл для airflow и скрипт для предобработки датасета.
    │   │
    │   ├── airflow-preprocessing-files     <- Докерфайл для airflow и скрипт для формирования датасета для работы с pandas.
    │   │
    │   ├── airflow-split                   <- Докерфайл для airflow и скрипт для разбиения на train/test выборки.
    │   │
    │   ├── airflow-train                   <- Докерфайл для airflow и скрипт для обучение модели.
    │   │
    │   ├── airflow-validation              <- Докерфайл для airflow и скрипт для получения метрик.
    │
    ├── docker-compose.yml                  <- Сохраненные модели, трансформеры, метрики, предсказания.
------------