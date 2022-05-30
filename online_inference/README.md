# Домашнее задание 2 по курсу Машинное обучение в продакшене

online inference (работа с docker и FastAPI)

#### Репозиторий в hub.docker

https://hub.docker.com/repository/docker/lllfantomlll/lllfantomlll/general

#### Собрать контейнер из директории online inference
~~~
docker build -t lllfantomlll:ready .
~~~

#### Загрузка образа из ретозитория hub.docker из директории online inference
~~~
docker pull lllfantomlll/lllfantomlll:ready
~~~

#### Запуск контейнера с docker образом из директории online inference
~~~
В одном окне консоли:
docker run --rm -p 80:80 lllfantomlll/lllfantomlll:ready
В другом окне консоли:
python make_predict_request.py
~~~

#### Запуск приложения из директории online inference
~~~
В одном окне консоли:
python app.py
В другом окне консоли:
python make_predict_request.py
~~~


Организация проекта
------------
    ├── data
    │   ├── raw.csv             <- Реальные данные.
    │   └── test_data.csv       <- Данные для теста.
    │
    ├── models                  <- Сохраненные модели, трансформеры.
    │
    ├── source                  <- Папка с классами для данных.
    │
    ├── tests                   <- Тесты.
    │
    ├── app.py                  <- Реализация общения с моделью через FastAPI.
    │
    ├── Dockerfile              <- Файл с настройками для docker.
    │
    ├── envinit.sh              <- Файл с настройками для виртуального окружения.
    │
    ├── make_predict_request    <- Скрипт для отправки команды предсказания.
    │
    ├── README.md               <- Инструкция к проекту.
    │
    ├── requirements.txt        <- Необходимые библиотеки для работы проекта.

--------