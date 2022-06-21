FROM python:3.8.10-slim

MAINTAINER lllfantomlll

ENV PATH_TO_MODEL="models/XGBClassifier.pkl"

COPY requirements.txt\
     app.py\
     envinit.sh ./

COPY source ./source
COPY  models ./models

RUN ./envinit.sh

WORKDIR .

EXPOSE 80

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
