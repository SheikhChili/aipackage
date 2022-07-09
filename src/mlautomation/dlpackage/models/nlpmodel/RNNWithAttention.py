from keras_tuner import HyperModel
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras import Sequential
from aipackage.dlpackage.models.ModelUtils import ModelUtils


# -----------------------------------------------RNN Attention Bidirectional LSTM Model--------------------------------------------------
class RNNAttentionBidirectionalLSTMModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ----------------------------------------- RNN Attention BIDIRECTIONAL GRU MODEL---------------------------------
class RNNAttentionBidirectionalGRUModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ---------------------------------------------RNN ATTENTION LSTM MODEL-------------------------------------------
class RNNAttentionLSTMModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# -----------------------------------------------RNN Attention GRU MODEL------------------------------------------
class RNNAttentionGRUModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ---------------------------------------------------RNNAttention Bidirectional LSTM AND LSTM Model--------------------------------------
class RNNAttentionBidirectionalLSTMANDLSTMModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# -------------------------------------------------RNN Attention BIDIRECTIONAL GRU AND LSTM MODEL----------------------------------------
class RNNAttentionBidirectionalGRUANDLSTMModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ------------------------------------------------------RNN Attention Bidirectional LSTM AND GRU Model-----------------------------------
class RNNAttentionBidirectionalLSTMANDGRUModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model


# ------------------------------------------------------RNN Attention BIDIRECTIONAL GRU AND GRU MODEL------------------------------------
class RNNAttentionBidirectionalGRUANDGRUModel(HyperModel):

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
			model.add(utils.get_nlp_attention_layer())
			model.add(utils.get_rnn_dropout_layer(hp, i))
		model.add(Flatten())
		model.add(utils.get_dense_layer(self.modelEntity.target_size, self.modelEntity.final_activation))
		                          
		model.summary()
		model.compile(optimizer=self.modelEntity.optimizer, loss=self.modelEntity.loss, metrics=self.modelEntity.metrics)
		return model
