from keras_tuner import HyperModel
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
from aipackage.dlpackage.models.ModelUtils import ModelUtils


# --------------------------------------ARTIFICIAL NEURAL NETWORK Model----------------------------------------------
class ANNDenseModel(HyperModel):

    def __init__(self):
        self.input_length = None
        self.modelEntity = None

    def set_model_data(self, model_entity):
        self.modelEntity = model_entity

    def set_input_shape(self, input_length):
        self.input_length = input_length

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_ann_input_layer(self.input_length))
        for i in range(utils.get_ann_range()):
            model.add(utils.get_ann_dense_layer(hp, i))
            model.add(utils.get_ann_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model
