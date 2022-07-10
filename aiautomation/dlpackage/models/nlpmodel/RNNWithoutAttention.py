from keras_tuner import HyperModel
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, BatchNormalization
from aipackage.dlpackage.models.ModelUtils import ModelUtils


# --------------------------------------Bidirectional LSTM Model----------------------------------------------
class BidirectionalLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_bi_lstm_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ---------------------------------------BIDIRECTIONAL GRU MODEL----------------------------------------------
class BidirectionalGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_bi_gru_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# --------------------------------------------------------LSTM MODEL--------------------------------------------------------------------
class LSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity
		
	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
			
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_lstm_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# -----------------------------------------------GRU MODEL----------------------------------------------------
class GRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity
	
	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
			
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_gru_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# -----------------------------------Bidirectional LSTM AND LSTM Model----------------------------------------
class BidirectionalLSTMANDLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity
	
	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
			
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_bi_lstm_layer(hp, i))
			model.add(utils.get_lstm_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ----------------------------------------BIDIRECTIONAL GRU AND LSTM MODEL------------------------------------
class BidirectionalGRUANDLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity
	
	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
			
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_bi_gru_layer(hp, i))
			model.add(utils.get_lstm_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# -----------------------------------------Bidirectional LSTM AND GRU Model----------------------------------
class BidirectionalLSTMANDGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity
		
	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
			
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_bi_lstm_layer(hp, i))
			model.add(utils.get_gru_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ----------------------------------------BIDIRECTIONAL GRU AND GRU MODEL-------------------------------------
class BidirectionalGRUANDGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity
	
	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
			
	def build(self, hp):
		utils = ModelUtils()
		
		model = Sequential()
		model.add(utils.get_nlp_embedding_layer(self.embeddingEntity))
		for i in range(utils.get_nlp_range()):
			model.add(utils.get_bi_gru_layer(hp, i))
			model.add(utils.get_gru_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model
