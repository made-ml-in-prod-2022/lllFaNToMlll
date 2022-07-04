import os
import datetime
import pickle
import json
import click
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


@click.command('validation')
@click.option('--validation_path')
@click.option('--metrics_path')
@click.option('--model_path')
def validate(validation_path: str, metrics_path: str, model_path: str):
    model = load_obj_pkl(os.path.join(model_path, 'model.pkl'))
    val = pd.read_csv(os.path.join(validation_path, 'test.csv'))
    y_test = val.target.values
    X_test = val.drop(['target'], axis=1).values
    y_pred = model.predict(X_test)
    accuracy_score = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {
        "Date": str(datetime.datetime.now()),
        "Roc|Auc": roc_auc,
        "Accuracy": accuracy_score,
        "F1": f1
    }

    os.makedirs(metrics_path, exist_ok=True)

    with open(os.path.join(metrics_path, 'metrics.json'), 'w') as file:
        json.dump(metrics, file)


def load_obj_pkl(path: str):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
        return obj


if __name__ == "__main__":
    validate()
