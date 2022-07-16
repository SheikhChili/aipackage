# IMPORT
import numpy as np
from sklearn.preprocessing import LabelEncoder
from aiautomation.mlpackage.Repository import Repository
from aiautomation.mlpackage.PackageVariable import Variable
from aiautomation.mlpackage.Entities import HyperParamEntity
from aiautomation.mlpackage.multiclassification.MultiClassModel import Models
from sklearn.model_selection import RepeatedStratifiedKFold
from aiautomation.mlpackage.HyperparameterTuning import HyperParameterTuning


class MultiClassifyRepository(Repository):

    def __init__(self):
        super().__init__()
        super().set_type(Variable.typeMultiClass)
        self.labels = []

    def process_data_and_run(self, train, label_name, df=None):
        train, df = super().process_and_get_train_data(train, label_name, df)

        # Fill nan and check
        print("COLUMN NULL CHECK")
        print(train.isnull().sum(), "\n")

        smote_x_train, smote_y_train = super().synthesize_data(train, df, label_name)

        x = np.array(smote_x_train.values.tolist())
        y = np.array(smote_y_train.values.tolist()).ravel()

        y = self.get_transformed_y(y)

        self.start_train(x, y)

    def get_transformed_y(self, y):
        print("Transformed Y start -----")
        if isinstance(y[0], str):
            pass

        le = LabelEncoder()
        le.fit(y)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        self.labels = [str(i) for i in le.classes_]

        print("Y before transformed -----------")
        print(y)
        print("Dictionary of label -----------")
        print(le_name_mapping)
        y = le.transform(y)
        print("Y after transformed -----------")
        print(y)
        print("LABELS LIST -------")
        print(self.labels)
        print("\n\n")
        return y

    def update_user_scoring_dict(self, user_scoring_dict=None):
        super().update_user_scoring_dict(user_scoring_dict)

    def start_train(self, x, y):
        super().update_xy_data(x, y)
        x_train, x_val, y_train, y_val = super().get_splitted_data()
        self.run_all_models(x_train, x_val, y_train, y_val)

    def concat_dataset_location(self):
        super().concat_to_dataset_location(Variable.typeMultiClass)

    def run_all_models(self, x_train, x_val, y_train, y_val):
        models = Models()
        model_array = models.get_all_models()

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        hp_entity = HyperParamEntity(x_train, y_train, x_val, y_val, models.get_algorithm_name(), cv,
                                     Variable.typeMultiClass, labels=self.labels)

        hyper_parameter_tuning = HyperParameterTuning(hp_entity)

        super().run_all_models(hyper_parameter_tuning, model_array)
