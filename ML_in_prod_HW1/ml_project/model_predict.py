import logging
import sys
import click
import pandas as pd
from parameters.predict_params import PredictParams, read_predict_params
from load_save_data.dataset_operations import read_data
from preprocessing_dataset import drop_target, transform_dataset
from model.model_fit_predict_save_load import predict_model, load_model, load_transformer


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_predict(config_path: str):
    logger.info('Запуск предсказания модели')
    logger.info('Подготовка данных')

    predict_params: PredictParams = read_predict_params(config_path)

    data = read_data(predict_params.input_data_path)

    if predict_params.target_in_dataset:
        data = drop_target(data, predict_params)

    transformer = load_transformer(predict_params.transformer_path)

    feature = transform_dataset(transformer, data)

    model = load_model(predict_params.model_path)

    logger.info('Предсказание')
    predict = predict_model(model, feature)

    logger.info('Запись предсказания в файл')
    pd.Series(predict, index=data.index, name='Predict').to_csv(predict_params.predict_path)


@click.command(name='run_predict')
@click.argument('config_path')
def run_predict_command(config_path: str):
    run_predict(config_path)


if __name__ == '__main__':
    run_predict_command()