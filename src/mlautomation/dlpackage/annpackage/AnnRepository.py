# import
from aipackage.dlpackage.Repository import Repository
from aipackage.dlpackage.annpackage.ANNModel import ANNModel
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.Entities import HyperParameterEntity
from aipackage.dlpackage.HyperParameterTuning import HyperParameterTuning


class AnnRepository(Repository):

    def __init__(self):
        super().__init__()

    def create_and_run_model(self, entities):
        annModel = ANNModel()
        # super().clearAfterConvertWordNumericVariable()
        self.run_all_models(annModel.get_ann_dense_model(), entities[0], entities[1])

    def run_all_models(self, model_list, model_entity, dataset_entity):
        print("RUN_ALL_MODELS\n")
        input_length = dataset_entity.X.shape[1]

        # self.clearAfterTrainStartVariable()
        hyperParameterTuning = HyperParameterTuning(dataset_entity)
        scores = super().get_max_score_and_val_score()
        hyperParameterTuning.set_scores(scores[0], scores[1])
        hyperParameterTuning.set_type(super().get_type())
        super().update_stored_model_file_name_array()
        for i in range(0, len(model_list)):
            model = model_list[i]
            model_entity.modelName = model.__class__.__name__.replace("Model", "")
            model.set_model_data(model_entity)
            model.set_input_shape(input_length)
            hyperParameterEntity = HyperParameterEntity(model, i)
            hyperParameterEntity.modelClassName = model.__class__.__name__ + str(Variable.ann_layer_count)
            self.run_model(hyperParameterEntity, hyperParameterTuning)

    def run_model(self, hyper_parameter_entity, hyper_parameter_tuning):
        super().run_model(hyper_parameter_entity, hyper_parameter_tuning)
