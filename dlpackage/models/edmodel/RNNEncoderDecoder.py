#IMPORTS
from keras_tuner import HyperModel
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.models.ModelUtils import ModelUtils


class EncoderDecoderBidirectionalLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', 0)
		encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_lstm_num_layers')):
			bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
				 	
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_lstm_units * 2, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state =encoderOutputs[1:])
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_lstm_num_layers')):
			decoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		

#----------------------------------------ENCODER BidirecLSTM AND DECODER LSTM MODEL----------------------------------------------------
class EncoderBidirectionalLSTMDecoderLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
	
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', 0)
		encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1]+encoderOutputs[3], encoderOutputs[2]+encoderOutputs[4]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_lstm_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_lstm_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_lstm_num_layers')):
			decoderOutputs = utils.get_ed_lstm_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		

#----------------------------------------ENCODER BidirecLSTM AND DECODER BidirecGRU MODEL-----------------------------------------------
class EncoderBidirectionalLSTMDecoderBidirectionalGRUModel(HyperModel):		

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', 0)
		encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1], encoderOutputs[3]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_lstm_units * 2, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_gru_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state = encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_gru_num_layers')):
			decoderOutputs = utils.get_ed_bi_gru_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(0))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		

#----------------------------------------ENCODER BidirecLSTM AND DECODER GRU MODEL-----------------------------------------------------
class EncoderBidirectionalLSTMDecoderGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
			
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', 0)
		encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			bi_lstm_units = utils.get_rnn_units(hp, 'en_bi_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_lstm_layer(bi_lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1]+encoderOutputs[3]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_lstm_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_gru_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_gru_num_layers')):
			decoderOutputs = utils.get_ed_gru_layer(bi_lstm_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		
								
#-------------------------------------ENCODER AND DECODER SAME BIDIRECTIONAL GRU MODEL-------------------------------------------------
class EncoderDecoderBidirectionalGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', 0)
		encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_gru_units * 2, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state =encoderOutputs[1:])
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_gru_num_layers')):
			decoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(0))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
			

#----------------------------------------ENCODER BidirecGRU AND DECODER BidirecLSTM MODEL-----------------------------------------------
class EncoderBidirectionalGRUDecoderBidirectionalLSTMModel(HyperModel):		#THROWING ERROR

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', 0)
		encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1], encoderOutputs[1], encoderOutputs[2], encoderOutputs[2]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_gru_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_lstm_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state = encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_lstm_num_layers')):
			decoderOutputs = utils.get_ed_bi_lstm_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)



#----------------------------------------ENCODER BidirecGRU AND DECODER LSTM MODEL------------------------------------------------
class EncoderBidirectionalGRUDecoderLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', 0)
		encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1]+encoderOutputs[2], encoderOutputs[1]+encoderOutputs[2]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_gru_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_lstm_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_lstm_num_layers')):
			decoderOutputs = utils.get_ed_lstm_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		
	
#----------------------------------------ENCODER BidirecGRU AND DECODER GRU MODEL------------------------------------------------
class EncoderBidirectionalGRUDecoderGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', 0)
		encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			bi_gru_units = utils.get_rnn_units(hp, 'en_bi_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_bi_gru_layer(bi_gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1]+encoderOutputs[2]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, bi_gru_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_gru_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_gru_num_layers')):
			decoderOutputs = utils.get_ed_gru_layer(bi_gru_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		
				
#-------------------------------------   ENCODER AND DECODER SAME LSTM MODEL------------------------------------------------------------
class EncoderDecoderLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', 0)
		encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, lstm_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoderOutputs[1:])
		for i in range(utils.get_decoder_range(hp, 'de_atn_lstm_num_layers')):
			decoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
	
	
	
#----------------------------------------ENCODER LSTM AND DECODER GRU MODEL-------------------------------------------------------------
class EncoderLSTMDecoderGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', 0)
		encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_lstm_num_layers')):
			lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, lstm_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_gru_layer(lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoderOutputs[1])
		for i in range(utils.get_decoder_range(hp, 'de_atn_gru_num_layers')):
			decoderOutputs = utils.get_ed_gru_layer(lstm_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])

		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		
		
#----------------------------------------ENCODER LSTM AND DECODER Bidireclstm MODEL-----------------------------------------------------
class EncoderLSTMDecoderBidirectionalLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', 0)
		encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_lstm_num_layers')):
			lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1], encoderOutputs[2], encoderOutputs[1], encoderOutputs[2]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, lstm_units * 2, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_lstm_layer(lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state = encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_lstm_num_layers')):
			decoderOutputs = utils.get_ed_bi_lstm_layer(lstm_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		
		
#----------------------------------------ENCODER LSTM AND DECODER BidirecGRU MODEL-----------------------------------------------------
class EncoderLSTMDecoderBidirectionalGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', 0)
		encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_lstm_num_layers')):
			lstm_units = utils.get_rnn_units(hp, 'en_lstm_units_', (i + 1))
			encoderOutputs = utils.get_ed_lstm_layer(lstm_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1], encoderOutputs[1]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, lstm_units * 2, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_gru_layer(lstm_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state = encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_gru_num_layers')):
			decoderOutputs = utils.get_ed_bi_gru_layer(lstm_units, Variable.decoder_rnn_prefix + str(0))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		
								
#----------------------------------------ENCODER AND DECODER SAME GRU MODEL-------------------------------------------------------------
class EncoderDecoderGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity
		
	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		gru_units = utils.get_rnn_units(hp, 'en_gru_units_', 0)
		encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_bi_gru_num_layers')):
			gru_units = utils.get_rnn_units(hp, 'en_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, gru_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoderOutputs[1:])
		for i in range(utils.get_decoder_range(hp, 'de_atn_gru_num_layers')):
			decoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)


		
#----------------------------------------ENCODER GRU AND DECODER LSTM MODEL-------------------------------------------------------------
class EncoderGRUDecoderLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		gru_units = utils.get_rnn_units(hp, 'en_gru_units_', 0)
		encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_gru_num_layers')):
			gru_units = utils.get_rnn_units(hp, 'en_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1], encoderOutputs[1]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, gru_units, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_lstm_layer(gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state=encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_lstm_num_layers')):
			decoderOutputs = utils.get_ed_lstm_layer(gru_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		


#----------------------------------------ENCODER GRU AND DECODER bidirec LSTM MODEL------------------------------------------------
class EncoderGRUDecoderBidirectionalLSTMModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		gru_units = utils.get_rnn_units(hp, 'en_gru_units_', 0)
		encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_gru_num_layers')):
			gru_units = utils.get_rnn_units(hp, 'en_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1], encoderOutputs[1], encoderOutputs[1], encoderOutputs[1]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, gru_units * 2, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_lstm_layer(gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state = encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_lstm_num_layers')):
			decoderOutputs = utils.get_ed_bi_lstm_layer(gru_units, Variable.decoder_rnn_prefix + str(i + 1))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
		
		
				
#----------------------------------------ENCODER GRU AND DECODER bidirecgru LSTM MODEL--------------------------------------------------
class EncoderGRUDecoderBidirectionalGRUModel(HyperModel):

	def setModelData(self, modelEntity):
		self.modelEntity = modelEntity

	def setEmbeddingData(self, embeddingEntity):
		self.embeddingEntity = embeddingEntity

	def build(self, hp):
		utils = ModelUtils()
		
		enInpLayer = utils.get_encoder_input_layer(self.embeddingEntity.input_length)
		enEmbInpLayer = utils.get_encoder_embedding_layer(self.embeddingEntity)(enInpLayer)
		gru_units = utils.get_rnn_units(hp, 'en_gru_units_', 0)
		encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(0))(enEmbInpLayer)
		for i in range(utils.get_encoder_range(hp, 'en_atn_gru_num_layers')):
			gru_units = utils.get_rnn_units(hp, 'en_gru_units_', (i + 1))
			encoderOutputs = utils.get_ed_gru_layer(gru_units, Variable.encoder_rnn_prefix + str(i + 1))(encoderOutputs[0])
		encoder_states = [encoderOutputs[1], encoderOutputs[1]]
		
		################################################ DECODER START #####################################################
		deInpLayer = utils.get_decoder_input_layer()
		deEmbInpLayer = utils.get_de_embed_layer(self.modelEntity.target_size, gru_units * 2, self.embeddingEntity.input_length)(deInpLayer)
		decoderOutputs = utils.get_ed_bi_gru_layer(gru_units, Variable.decoder_rnn_prefix + str(0))(deEmbInpLayer, initial_state = encoder_states)
		for i in range(utils.get_decoder_range(hp, 'de_atn_bi_gru_num_layers')):
			decoderOutputs = utils.get_ed_bi_gru_layer(gru_units, Variable.decoder_rnn_prefix + str(0))(decoderOutputs[0])
                  
		return utils.process_and_get_ed_model(enInpLayer, deInpLayer, encoderOutputs[0], decoderOutputs[0], self.modelEntity)
