# IMPORT
import numpy as np
from sklearn.preprocessing import LabelEncoder
from aiautomation.mlpackage.Entities import ModelVisualizeEntity
from aiautomation.mlpackage.Repository import Repository
from aiautomation.mlpackage.PackageVariable import Variable
from aiautomation.mlpackage.Entities import HyperParamEntity
from aiautomation.mlpackage.classification.ClassificationModel import Models
from sklearn.model_selection import RepeatedStratifiedKFold
from aiautomation.mlpackage.HyperparameterTuning import HyperParameterTuning


class ClassifyRepository(Repository):

    def __init__(self):
        super().__init__()
        super().set_type(Variable.typeClassification)
        self.labels = []

    def process_data_and_run(self, train, label_name, df=None):
        train, df = super().process_and_get_train_data(train, label_name, df)

        # Fill nan and check
        print("COLUMN NULL CHECK TRAIN")
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

        self.labels = le.classes_

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
        super().concat_to_dataset_location(Variable.typeClassification)

    def run_all_models(self, x_train, x_val, y_train, y_val):
        models = Models()
        model_array = models.get_all_models(x_train.shape[1])

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        print(self.labels)
        hp_entity = HyperParamEntity(x_train, y_train, x_val, y_val, models.get_algorithm_name(), cv,
                                     Variable.typeClassification, labels=self.labels)

        hyper_parameter_tuning = HyperParameterTuning(hp_entity)

        super().run_all_models(hyper_parameter_tuning, model_array)

    def run_all_models(self, x_train, x_val, y_train, y_val):
        models = Models()
        model_array = models.get_all_models(x_train.shape[1])

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        print(self.labels)
        hp_entity = HyperParamEntity(x_train, y_train, x_val, y_val, models.get_algorithm_name(), cv,
                                     Variable.typeClassification, labels=self.labels)

        hyper_parameter_tuning = HyperParameterTuning(hp_entity)

        super().run_all_models(hyper_parameter_tuning, model_array)

    def run_all_visualize_models(self, x_train, x_val, y_train, y_val):
        models = Models()
        saved_model_array = super().get_all_saved_model()

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        print(self.labels)
        hp_entity = HyperParamEntity(x_train, y_train, x_val, y_val, models.get_algorithm_name(), cv,
                                     Variable.typeClassification, labels=self.labels)

        hyper_parameter_tuning = HyperParameterTuning(hp_entity)

        super().run_all_models(hyper_parameter_tuning, self.create_model_visualize_entity_array(models,
                                                                                                saved_model_array),
                               should_run_search=False)

    @staticmethod
    def create_model_visualize_entity_array(models, saved_model_array, max_feature):
        saved_entity_array = []
        for model_path in saved_model_array:
            model = super().load_model(model_path)
            model_type = super().check_and_get_model_type(model_path)
            file_name = model_path.split(Variable.locationSeparator)[-1].removesuffix(Variable.pickleExtension)
            saved_entity_array.append(ModelVisualizeEntity(model, models.check_and_get_grid_entity(model_type, max_feature),
                                                           model_type, super().check_and_get_search_type(model),
                                                           file_name))
        return saved_entity_array
