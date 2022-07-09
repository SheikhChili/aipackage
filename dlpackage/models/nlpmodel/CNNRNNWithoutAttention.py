from keras_tuner import HyperModel
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,BatchNormalization
from aipackage.dlpackage.models.ModelUtils import ModelUtils


# ----------------------------------CNN Bidirectional LSTM Model-------------------------------------------------------
class CNNBidirectionalLSTMModel(HyperModel):

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
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ----------------------------------CNN BIDIRECTIONAL GRU MODEL---------------------------------------------------------
class CNNBidirectionalGRUModel(HyperModel):

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
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ----------------------------------CNN LSTM MODEL----------------------------------------------------------------------
class CNNLSTMModel(HyperModel):

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
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ----------------------------------CNN GRU MODEL-----------------------------------------------------------------------
class CNNGRUModel(HyperModel):

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
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ---------------------------------CNN Bidirectional LSTM AND LSTM Model------------------------------------------------
class CNNBidirectionalLSTMANDLSTMModel(HyperModel):

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
			model.add(utils.get_lstm_layer(hp, i))
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ---------------------------------CNN BIDIRECTIONAL GRU AND LSTM MODEL-------------------------------------------------
class CNNBidirectionalGRUANDLSTMModel(HyperModel):

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
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ---------------------------------CNN Bidirectional LSTM AND GRU Model-------------------------------------------------
class CNNBidirectionalLSTMANDGRUModel(HyperModel):

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
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# --------------------------------CNN BIDIRECTIONAL GRU AND GRU MODEL---------------------------------------------------
class CNNBidirectionalGRUANDGRUModel(HyperModel):

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
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model
