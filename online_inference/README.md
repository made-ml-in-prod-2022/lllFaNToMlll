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