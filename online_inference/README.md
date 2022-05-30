# Домашнее задание 2 по курсу Машинное обучение в продакшене

online inference (работа с docker и FastAPI)

#### Запуск тестов из директории online inference

~~~
python -m pytest tests\tests.py
~~~

#### Загрузка образа из ретозитория hub.docker
~~~
docker pull lllfantomlll/lllfantomlll:ready
~~~

#### Запуск контейнера с docker образом
~~~
docker run --rm -p 80:80 lllfantomlll/lllfantomlll:ready
~~~

Организация проекта
------------
    ├── data
    │   ├── raw.csv             <- Реальные данные.
	│	└── test_data.csv       <- Данные для теста.
	│
    ├── models                  <- Сохраненные модели, трансформеры.
    │
	├── source              	<- Папка с классами для данных.
	│
	├── tests                   <- Тесты.
	│
	├── app.py                  <- Реализация общения с моделью через FastAPI.
    │
    ├── Dockerfile              <- Файл с настройками для docker.
	│
	├── envinit.sh              <- Файл с настройками для виртуального окружения.
	│
	├── make_request            <- Скрипт для отправки команды предсказания.
	│
	├── README.md               <- Инструкция к проекту.
	│
    ├── requirements.txt        <- Необходимые библиотеки для работы проекта.

--------