import os
import click
import numpy as np
from sklearn.datasets import load_breast_cancer


@click.command("generate")
@click.option("--out_path")
def generate_data_cancer(out_path: str):
    n_data = np.random.randint(50, 550)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    os.makedirs(out_path, exist_ok=True)
    X[:n_data].to_csv(os.path.join(out_path, 'data.csv'))
    y[:n_data].to_csv(os.path.join(out_path, 'target.csv'))
    print('###################################', y[:n_data])


if __name__ == '__main__':
    generate_data_cancer()
