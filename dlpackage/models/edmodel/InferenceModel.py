#IMPORTS
from tensorflow.keras.layers import Input
from aipackage.dlpackage.models.ModelUtils import ModelUtils


class EncoderDecoderBidirectionalLSTMInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpFC = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deStateInpBC = Input(shape=(units,))
		deIntialState = [deStateInpFH, deStateInpFC, deStateInpBH, deStateInpBC]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)
		
		
		
#-------------------------------------ENCODER AND DECODER SAME BIDIRECTIONAL GRU MODEL-------------------------------------------------
class EncoderDecoderBidirectionalGRUInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deIntialState = [deStateInpFH,  deStateInpBH]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)
		


#-------------------------------------   ENCODER AND DECODER SAME LSTM MODEL------------------------------------------------------------
class EncoderDecoderLSTMInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deStateInpC = Input(shape=(units,))
		deIntialState = [deStateInpH, deStateInpC]
	
		return utils.get_inference_model(inferenceEntity, units, deIntialState)
		
		
		
#----------------------------------------ENCODER AND DECODER SAME GRU MODEL-------------------------------------------------------------
class EncoderDecoderGRUInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deIntialState = [deStateInpH]
	
		return utils.get_inference_model(inferenceEntity, units, deIntialState)
		


#----------------------------------------ENCODER BidirecLSTM AND DECODER BidirecGRU MODEL-----------------------------------------------
class EncoderBidirectionalLSTMDecoderBidirectionalGRUInferenceModel():		

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deIntialState = [deStateInpFH, deStateInpBH]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)
		

		
#----------------------------------------ENCODER BidirecGRU AND DECODER BidirecLSTM MODEL-----------------------------------------------
class EncoderBidirectionalGRUDecoderBidirectionalLSTMInferenceModel():		#THROWING ERROR

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpFC = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deStateInpBC = Input(shape=(units,))
		deIntialState = [deStateInpFH, deStateInpFC,deStateInpBH,deStateInpBC]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)
		

		
#----------------------------------------ENCODER BidirecLSTM AND DECODER LSTM MODEL----------------------------------------------------
class EncoderBidirectionalLSTMDecoderLSTMInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deStateInpC = Input(shape=(units,))
		deIntialState = [deStateInpH, deStateInpC]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)
		


#----------------------------------------ENCODER BidirecLSTM AND DECODER GRU MODEL-----------------------------------------------------
class EncoderBidirectionalLSTMDecoderGRUInferenceModel():
			
	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deIntialState = [deStateInpH]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)

			
		
#----------------------------------------ENCODER BidirecGRU AND DECODER LSTM MODEL------------------------------------------------
class EncoderBidirectionalGRUDecoderLSTMInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deStateInpC = Input(shape=(units,))
		deIntialState = [deStateInpH, deStateInpC]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)
		


#----------------------------------------ENCODER BidirecGRU AND DECODER GRU MODEL------------------------------------------------
class EncoderBidirectionalGRUDecoderGRUInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deIntialState = [deStateInpH]
		
		return utils.get_inference_model(inferenceEntity, units * 2, deIntialState)
		


#----------------------------------------ENCODER LSTM AND DECODER GRU MODEL-------------------------------------------------------------
class EncoderLSTMDecoderGRUInferenceModel():
		
	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deIntialState = [deStateInpH]
		
		return utils.get_inference_model(inferenceEntity, units, deIntialState)
		

		
#----------------------------------------ENCODER GRU AND DECODER LSTM MODEL-------------------------------------------------------------
class EncoderGRUDecoderLSTMInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpH = Input(shape=(units,))
		deStateInpC = Input(shape=(units,))
		deIntialState = [deStateInpH, deStateInpC]
		
		return utils.get_inference_model(inferenceEntity, units, deIntialState)


		
#----------------------------------------ENCODER LSTM AND DECODER Bidireclstm MODEL-----------------------------------------------------
class EncoderLSTMDecoderBidirectionalLSTMInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpFC = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deStateInpBC = Input(shape=(units,))
		deIntialState = [deStateInpFH, deStateInpFC,deStateInpBH,deStateInpBC]
		
		return utils.get_inference_model(inferenceEntity, units, deIntialState)


		
#----------------------------------------ENCODER LSTM AND DECODER BidirecGRU MODEL-----------------------------------------------------
class EncoderLSTMDecoderBidirectionalGRUInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deIntialState = [deStateInpFH, deStateInpBH]

		return utils.get_inference_model(inferenceEntity, units, deIntialState)


		
#----------------------------------------ENCODER GRU AND DECODER bidirecgru LSTM MODEL--------------------------------------------------
class EncoderGRUDecoderBidirectionalGRUInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deIntialState = [deStateInpFH, deStateInpBH]
		
		return utils.get_inference_model(inferenceEntity, units, deIntialState)

		
#----------------------------------------ENCODER GRU AND DECODER bidirec LSTM MODEL------------------------------------------------
class EncoderGRUDecoderBidirectionalLSTMInferenceModel():

	def getInferenceModel(self, inferenceEntity):
		utils = ModelUtils()
		
		units = inferenceEntity.units
		# Below tensors will hold the states of the previous time step
		deStateInpFH = Input(shape=(units,))
		deStateInpFC = Input(shape=(units,))
		deStateInpBH = Input(shape=(units,))
		deStateInpBC = Input(shape=(units,))
		deIntialState = [deStateInpFH, deStateInpFC,deStateInpBH, deStateInpBC]

		return utils.get_inference_model(inferenceEntity, units, deIntialState)
