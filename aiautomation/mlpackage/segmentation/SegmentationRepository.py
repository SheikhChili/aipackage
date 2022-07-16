# IMPORT
import numpy as np
from sklearn.preprocessing import LabelEncoder
from aiautomation.mlpackage.Repository import Repository
from aiautomation.mlpackage.PackageVariable import Variable
from aiautomation.mlpackage.Entities import HyperParamEntity
from aiautomation.mlpackage.segmentation.SegmentationModel import Models
from sklearn.model_selection import RepeatedStratifiedKFold
from aiautomation.mlpackage.HyperparameterTuning import HyperParameterTuning


class SGRepository(Repository):

    def __init__(self):
        super().__init__()
        super().set_type(Variable.typeSegmentation)

    def process_data_and_run(self, train, label_name, df=None):
        train, df = super().process_and_get_train_data(train, label_name, df)

        # Fill nan and check
        print("COLUMN NULL CHECK TRAIN")
        print(train.isnull().sum(), "\n")

        smote_x_train, smote_y_train = super().synthesize_data(train, df, label_name)

        x = np.array(smote_x_train.values.tolist())
        y = np.array(smote_y_train.values.tolist())

        if isinstance(y[0][0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        self.start_train(x, y)

    def update_user_scoring_dict(self, user_scoring_dict=None):
        super().update_user_scoring_dict(user_scoring_dict)

    def start_train(self, x, y):
        super().update_xy_data(x, y)
        x_train, x_val, y_train, y_val = super().get_splitted_data()
        self.run_all_models(x_train, x_val, y_train, y_val)

    def concat_dataset_location(self):
        super().concat_to_dataset_location(Variable.typeSegmentation)

    def run_all_models(self, x_train, x_val, y_train, y_val):
        models = Models(cluster=4)
        model_array = models.get_all_models()

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        hp_entity = HyperParamEntity(x_train, y_train, x_val, y_val, models.get_algorithm_name(), cv,
                                     Variable.typeSegmentation)

        hyper_parameter_tuning = HyperParameterTuning(hp_entity)

        super().run_all_models(hyper_parameter_tuning, model_array)
