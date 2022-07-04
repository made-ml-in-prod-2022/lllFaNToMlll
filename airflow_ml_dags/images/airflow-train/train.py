import os
import click
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--load_data_path")
@click.option("--save_model_path")
def train_model(load_data_path: str, save_model_path: str):
    data = pd.read_csv(os.path.join(load_data_path, 'train.csv'))
    y_train = data.target.values
    X_train = data.drop(['target'], axis=1).values
    model = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
    os.makedirs(save_model_path, exist_ok=True)
    save_object_pkl(model, os.path.join(save_model_path, 'model.pkl'))


def save_object_pkl(obj, path: str):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


if __name__ == "__main__":
    train_model()
