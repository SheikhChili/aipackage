from keras_tuner import HyperModel
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, BatchNormalization
from aipackage.dlpackage.models.ModelUtils import ModelUtils


# ----------------------------------CNN Attention Bidirectional LSTM Model---------------------------------------------
class CNNAttentionBidirectionalLSTMModel(HyperModel):

    def __init__(self):
        self.embeddingEntity = None
        self.modelEntity = None

    def set_model_data(self, model_entity):
        self.modelEntity = model_entity

    def set_embedding_data(self, embedding_entity):
        self.embeddingEntity = embedding_entity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_bi_lstm_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model


# -----------------------------------CNN Attention BIDIRECTIONAL GRU MODEL---------------------------------------------
class CNNAttentionBidirectionalGRUModel(HyperModel):

    def setModelData(self, modelEntity):
        self.modelEntity = modelEntity

    def setEmbeddingData(self, embeddingEntity):
        self.embeddingEntity = embeddingEntity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_bi_gru_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model


# ------------------------------------CNN ATTENTION LSTM MODEL----------------------------------------------------------
class CNNAttentionLSTMModel(HyperModel):

    def setModelData(self, modelEntity):
        self.modelEntity = modelEntity

    def setEmbeddingData(self, embeddingEntity):
        self.embeddingEntity = embeddingEntity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_lstm_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model


# ------------------------------------CNN Attention GRU MODEL-----------------------------------------------------------
class CNNAttentionGRUModel(HyperModel):

    def setModelData(self, modelEntity):
        self.modelEntity = modelEntity

    def setEmbeddingData(self, embeddingEntity):
        self.embeddingEntity = embeddingEntity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_gru_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model


# ------------------------------------CNN Attention Bidirectional LSTM AND LSTM Model-----------------------------------
class CNNAttentionBidirectionalLSTMANDLSTMModel(HyperModel):  # not working

    def setModelData(self, modelEntity):
        self.modelEntity = modelEntity

    def setEmbeddingData(self, embeddingEntity):
        self.embeddingEntity = embeddingEntity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_bi_lstm_layer(hp, i))
            model.add(utils.get_lstm_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model


# ------------------------------------CNN Attention BIDIRECTIONAL GRU AND LSTM MODEL------------------------------------
class CNNAttentionBidirectionalGRUANDLSTMModel(HyperModel):

    def setModelData(self, modelEntity):
        self.modelEntity = modelEntity

    def setEmbeddingData(self, embeddingEntity):
        self.embeddingEntity = embeddingEntity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_bi_gru_layer(hp, i))
            model.add(utils.get_lstm_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model


# ------------------------------------------CNN Attention Bidirectional LSTM AND GRU Model---------------------------------------------
class CNNAttentionBidirectionalLSTMANDGRUModel(HyperModel):

    def setModelData(self, modelEntity):
        self.modelEntity = modelEntity

    def setEmbeddingData(self, embeddingEntity):
        self.embeddingEntity = embeddingEntity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_bi_lstm_layer(hp, i))
            model.add(utils.get_gru_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model


# --------------------------------------- CNN Attention BIDIRECTIONAL GRU AND GRU MODEL-------------------------------------------------
class CNNAttentionBidirectionalGRUANDGRUModel(HyperModel):

    def setModelData(self, modelEntity):
        self.modelEntity = modelEntity

    def setEmbeddingData(self, embeddingEntity):
        self.embeddingEntity = embeddingEntity

    def build(self, hp):
        utils = ModelUtils()

        model = Sequential()
        model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
        for i in range(utils.get_nlp_range()):
            model.add(utils.get_conv_layer(hp, i))
            model.add(utils.get_max_pool_layer(hp, i))
            model.add(utils.get_cnn_dropout_layer(hp, i))

            model.add(utils.get_bi_gru_layer(hp, i))
            model.add(utils.get_gru_layer(hp, i))
            model.add(utils.get_nlp_attention_layer())
            model.add(utils.get_rnn_dropout_layer(hp, i))
        model.add(Flatten())
        model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))

        model.summary()
        model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss,
                      metrics=self.modelEntity.metrics)
        return model
