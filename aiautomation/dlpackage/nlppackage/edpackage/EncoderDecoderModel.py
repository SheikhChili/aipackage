
#IMPORTS
from aipackage.dlpackage.Models.EDModel.RNNEncoderDecoder import EncoderDecoderBidirectionalLSTMModel, EncoderDecoderBidirectionalGRUModel, 			EncoderDecoderLSTMModel, EncoderDecoderGRUModel, EncoderBidirectionalLSTMDecoderBidirectionalGRUModel, 		 			EncoderBidirectionalLSTMDecoderLSTMModel, EncoderBidirectionalLSTMDecoderGRUModel, EncoderBidirectionalGRUDecoderLSTMModel, 			EncoderBidirectionalGRUDecoderGRUModel, EncoderLSTMDecoderGRUModel, EncoderGRUDecoderLSTMModel, 			 			EncoderLSTMDecoderBidirectionalLSTMModel, EncoderLSTMDecoderBidirectionalGRUModel, EncoderGRUDecoderBidirectionalGRUModel, 			EncoderGRUDecoderBidirectionalLSTMModel, EncoderBidirectionalGRUDecoderBidirectionalLSTMModel
from aipackage.dlpackage.Models.EDModel.InferenceModel import EncoderDecoderBidirectionalLSTMInferenceModel, EncoderDecoderBidirectionalGRUInferenceModel, EncoderDecoderLSTMInferenceModel, EncoderDecoderGRUInferenceModel, EncoderBidirectionalLSTMDecoderBidirectionalGRUInferenceModel, EncoderBidirectionalGRUDecoderBidirectionalLSTMInferenceModel, EncoderBidirectionalLSTMDecoderLSTMInferenceModel, EncoderBidirectionalLSTMDecoderGRUInferenceModel, EncoderBidirectionalGRUDecoderLSTMInferenceModel, EncoderBidirectionalGRUDecoderGRUInferenceModel, EncoderLSTMDecoderGRUInferenceModel, EncoderGRUDecoderLSTMInferenceModel, EncoderLSTMDecoderBidirectionalLSTMInferenceModel, EncoderLSTMDecoderBidirectionalGRUInferenceModel, EncoderGRUDecoderBidirectionalGRUInferenceModel, EncoderGRUDecoderBidirectionalLSTMInferenceModel 
    

class EDModel():
	
	def getAllEDModels(self):
		return [EncoderDecoderBidirectionalLSTMModel(), EncoderDecoderBidirectionalGRUModel(), EncoderDecoderLSTMModel(), EncoderDecoderGRUModel(), EncoderBidirectionalLSTMDecoderBidirectionalGRUModel(), EncoderBidirectionalLSTMDecoderLSTMModel(), EncoderBidirectionalLSTMDecoderGRUModel(), EncoderBidirectionalGRUDecoderBidirectionalLSTMModel(), EncoderBidirectionalGRUDecoderLSTMModel(), EncoderBidirectionalGRUDecoderGRUModel(), EncoderLSTMDecoderGRUModel(),  EncoderLSTMDecoderBidirectionalLSTMModel(), EncoderLSTMDecoderBidirectionalGRUModel(), EncoderGRUDecoderBidirectionalGRUModel(), EncoderGRUDecoderBidirectionalLSTMModel(), EncoderGRUDecoderLSTMModel()]
             	
	def getAllInferenceModel(self):
		return [EncoderDecoderBidirectionalLSTMInferenceModel(), EncoderDecoderBidirectionalGRUInferenceModel(), EncoderDecoderLSTMInferenceModel(), EncoderDecoderGRUInferenceModel(), EncoderBidirectionalLSTMDecoderBidirectionalGRUInferenceModel(), EncoderBidirectionalGRUDecoderBidirectionalLSTMInferenceModel(), EncoderBidirectionalLSTMDecoderLSTMInferenceModel(), EncoderBidirectionalLSTMDecoderGRUInferenceModel(), EncoderBidirectionalGRUDecoderLSTMInferenceModel(), EncoderBidirectionalGRUDecoderGRUInferenceModel(), EncoderLSTMDecoderGRUInferenceModel(), EncoderGRUDecoderLSTMInferenceModel(), EncoderLSTMDecoderBidirectionalLSTMInferenceModel(), EncoderLSTMDecoderBidirectionalGRUInferenceModel(), EncoderGRUDecoderBidirectionalGRUInferenceModel(), EncoderGRUDecoderBidirectionalLSTMInferenceModel()
		] 
		
		
	def getInferenceModelClass(self, name):
		print(name)
		for model in self.getAllInferenceModel():
			if name in model.__class__.__name__:
				return model	            	
		
