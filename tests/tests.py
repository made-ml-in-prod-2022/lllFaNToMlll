import unittest
from ml_project.load_save_data import read_data, split_train_test_data
from ml_project.model.model_fit_predict_save_load import load_transformer, load_model, predict_model
from ml_project.parameters.dataset_params import SplittingParams
from ml_project.preprocessing_dataset.preprocessing_dataset import transform_dataset


class TestProject(unittest.TestCase):
    def test_predict_model(self):
        data = read_data('C:/Users/miair/PycharmProjects/ML_in_prod_HW1/data/raw.csv')
        data = data.drop(columns=['condition'])
        transformer = load_transformer("C:/Users/miair/PycharmProjects/ML_in_prod_HW1/models/XGBClassifier_transformer.pkl")
        feature = transform_dataset(transformer, data)
        model = load_model("C:/Users/miair/PycharmProjects/ML_in_prod_HW1/models/XGBClassifier.pkl")
        predict = predict_model(model, feature)
        self.assertEqual(297, len(predict))

    def test_read_data(self):
        data = read_data('C:/Users/miair/PycharmProjects/ML_in_prod_HW1/data/raw.csv')
        self.assertEqual(297, len(data))

    def test_split_data(self):
        data = read_data('C:/Users/miair/PycharmProjects/ML_in_prod_HW1/data/raw.csv')
        splitting_params = SplittingParams(random_state=42, test_size=0.2)
        train, test = split_train_test_data(data, splitting_params)
        self.assertEqual(237, len(train))
        self.assertEqual(60, len(test))


if __name__ == '__main__':
    unittest.main()
