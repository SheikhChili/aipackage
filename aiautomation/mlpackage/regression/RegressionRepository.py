# IMPORT
import numpy as np
from sklearn.model_selection import KFold
from aiautomation.mlpackage.Repository import Repository
from aiautomation.mlpackage.regression.RegressionModel import Models
from aiautomation.mlpackage.PackageVariable import Variable
from aiautomation.mlpackage.Entities import HyperParamEntity
from aiautomation.mlpackage.HyperparameterTuning import HyperParameterTuning


class RegRepository(Repository):

    def __init__(self):
        super().__init__()
        super().set_type(Variable.typeRegress)

    def process_data_and_run(self, train, label_name, df=None):
        train, df = super().process_and_get_train_data(train, label_name, df)

        # Fill nan and check
        print("COLUMN NULL CHECK TRAIN")
        print(train.isnull().sum(), "\n")

        x = np.array(train.values.tolist())
        y = np.array(df.values.tolist())

        print(x.shape)
        print(y.shape, "\n")
        self.start_train(x, y)

    def update_user_scoring_dict(self, user_scoring_dict=None):
        super().update_user_scoring_dict(user_scoring_dict)

    def start_train(self, x, y):
        super().update_xy_data(x, y)
        x_train, x_val, y_train, y_val = super().get_splitted_data()
        self.run_all_models(x_train, y_train, x_val, y_val)

    def concat_dataset_location(self):
        super().concat_to_dataset_location(Variable.typeRegress)

    def run_all_models(self, x_train, y_train, x_val, y_val):
        models = Models()
        model_array = models.get_all_models()

        cv = KFold(n_splits=10, shuffle=True, random_state=100)
        hp_entity = HyperParamEntity(x_train, y_train, x_val, y_val, models.get_algorithm_name(), cv,
                                     Variable.typeRegress)

        hyper_parameter_tuning = HyperParameterTuning(hp_entity)

        super().run_all_models(hyper_parameter_tuning, model_array)
