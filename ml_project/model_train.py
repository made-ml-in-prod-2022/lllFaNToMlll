"""
Файл для обучения модели.
На вход подается конфиг для обучения модели.
"""
import json
import logging
import sys
import click
import pandas as pd


from parameters.train_params import TrainingParams, read_training_params

from load_save_data.dataset_operations import read_data, split_train_test_data

from preprocessing_dataset.preprocessing_dataset import \
    extract_target, drop_target, build_feature_transformer, transform_dataset

from preprocessing_dataset.custom_transformer import CustomTransformer

from model.model_fit_predict_save_load import \
    train_model, predict_model, calculate_metrics, save_model, save_transformer


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_train(config_path: str):
    """Главная функция, запускающая весь процесс обучения"""
    training_params: TrainingParams = \
        read_training_params(config_path)

    if training_params.model_params.model_type == 'XGBClassifier':
        logger.info(f'Запуск обучения модели {training_params.model_params.model_type}'
                    f' с параметрами: \n'
                    f'random_state = {training_params.model_params.random_state}\n'
                    f'learning_rate = {training_params.model_params.learning_rate}\n'
                    f'max_depth = {training_params.model_params.max_depth}\n'
                    f'n_estimators = {training_params.model_params.n_estimators}')

    elif training_params.model_params.model_type == 'LogisticRegression':
        logger.info(f'Запуск обучения модели {training_params.model_params.model_type}'
                    f' с параметрами: \n'
                    f'random_state = {training_params.model_params.random_state}\n'
                    f'learning_rate = {training_params.model_params.penalty}\n'
                    f'max_depth = {training_params.model_params.max_iter}\n'
                    f'n_estimators = {training_params.model_params.solver}')

    data: pd.DataFrame = read_data(training_params.input_data_path)

    train_df, valid_df = split_train_test_data(
        data, training_params.splitting_params
    )

    logger.info('Разбиение на обучающую и тестовую выборки')

    train_target = extract_target(
        train_df, training_params.feature_params
    )
    train_df = drop_target(
        train_df, training_params.feature_params
    )

    valid_target = extract_target(
        valid_df, training_params.feature_params
    )
    valid_df = drop_target(
        valid_df, training_params.feature_params
    )

    logger.info(f'Размерность обучающего датасета: {train_df.shape}')
    logger.info(f'Размерность тестового датасета: {valid_df.shape}')

    if training_params.custom_transformer_params.use_custom_transformer:
        transformer = CustomTransformer(training_params.feature_params)
        transformer.fit(data)
    else:
        transformer = build_feature_transformer(training_params.feature_params)
        transformer.fit(train_df)

    save_transformer(transformer, training_params.save_transformer)

    train_features = transform_dataset(transformer, train_df)

    model = train_model(train_features, train_target, training_params.model_params)

    valid_feature = transform_dataset(transformer, valid_df)

    predict = predict_model(model, valid_feature)

    metrics = calculate_metrics(predict, valid_target, training_params.save_model)

    logger.info(f'Значения метрик: {metrics}')

    with open(training_params.metric_path, 'w', encoding='utf-8') as file_metrics:
        json.dump(metrics, file_metrics)

    save_model(model, training_params.save_model)


@click.command(name='run_train')
@click.argument('config_path', default='configs/train_config.yaml')
def train_command(config_path: str = "configs/XGBClassifier_train_config.yaml"):
    """функция для запуска процесса обучения"""
    run_train(config_path)


if __name__ == '__main__':
    train_command()
