# IMPORT
import numpy as np
from sklearn.model_selection import KFold
from aipackage.mlpackage.Repository import Repository
from aipackage.regression.RegressionModel import Models
from aipackage.mlpackage.PackageVariable import Variable
from aipackage.mlpackage.Entities import HyperParamEntity
from aipackage.mlpackage.HyperparameterTuning import HyperParameterTuning


class RegRepository(Repository):

    def __init__(self):
        super().__init__()
        super().set_type(Variable.typeRegress)

    def process_data_and_run(self, train, label_name, df=None):
        train, df = super().process_and_get_train_data(train, label_name, df)

        # Fill nan and check
        print("COLUMN NULL CHECK TRAIN")
        print(train.isnull().sum(), "\n")

        X = np.array(train.values.tolist())
        Y = np.array(df.values.tolist())

        print(X.shape)
        print(Y.shape, "\n")
        self.start_train(X, Y)

    def update_user_scoring_dict(self, user_scoring_dict=None):
        super().update_user_scoring_dict(user_scoring_dict)

    def start_train(self, x, y):
        super().update_xy_data(x, y)
        x_train, x_val, y_train, y_val = super().get_splited_data()
        self.run_all_models(x_train, y_train, x_val, y_val)

    def concat_dataset_location(self):
        super().concat_to_dataset_location(Variable.typeRegress)

    def run_all_models(self, x_train, y_train, x_val, y_val):
        models = Models()
        modelArray = models.get_all_models()

        cv = KFold(n_splits=10, shuffle=True, random_state=100)
        hp_entity = HyperParamEntity(x_train, y_train, x_val, y_val, models.get_algorithm_name(), cv,
                                     Variable.typeRegress)

        hyper_parameter_tuning = HyperParameterTuning(hp_entity)

        super().run_all_models(hyper_parameter_tuning, modelArray)
