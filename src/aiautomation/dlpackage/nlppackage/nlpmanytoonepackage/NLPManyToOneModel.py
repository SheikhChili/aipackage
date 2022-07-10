
#IMPORTS
from aipackage.dlpackage.models.nlpmodel.CNNRNNWithAttention import CNNAttentionBidirectionalLSTMModel, CNNAttentionBidirectionalGRUModel, 			CNNAttentionLSTMModel, CNNAttentionGRUModel, CNNAttentionBidirectionalLSTMANDLSTMModel, 				 			CNNAttentionBidirectionalGRUANDLSTMModel, CNNAttentionBidirectionalLSTMANDGRUModel, CNNAttentionBidirectionalGRUANDGRUModel
from aipackage.dlpackage.models.nlpmodel.CNNRNNWithoutAttention import CNNBidirectionalLSTMModel, CNNBidirectionalGRUModel, CNNLSTMModel, 			CNNGRUModel, CNNBidirectionalLSTMANDLSTMModel, CNNBidirectionalGRUANDLSTMModel, CNNBidirectionalLSTMANDGRUModel, 			CNNBidirectionalGRUANDGRUModel
from aipackage.dlpackage.models.nlpmodel.RNNWithAttention import RNNAttentionBidirectionalLSTMModel, RNNAttentionBidirectionalGRUModel, 			RNNAttentionLSTMModel, RNNAttentionGRUModel, RNNAttentionBidirectionalLSTMANDLSTMModel, 			         			RNNAttentionBidirectionalGRUANDLSTMModel, RNNAttentionBidirectionalLSTMANDGRUModel
from aipackage.dlpackage.models.nlpmodel.RNNWithoutAttention import BidirectionalLSTMModel, BidirectionalGRUModel, LSTMModel, GRUModel, 		 	BidirectionalLSTMANDLSTMModel, BidirectionalGRUANDLSTMModel, BidirectionalLSTMANDGRUModel, 				 			BidirectionalGRUANDGRUModel
    
  
class NLPManyToOneModel():
	
	def getAllNLPManyToOneModels(self):
		return [
			BidirectionalLSTMModel()#, BidirectionalGRUModel(), LSTMModel(), GRUModel(), BidirectionalLSTMANDLSTMModel(), 				BidirectionalGRUANDLSTMModel(), BidirectionalLSTMANDGRUModel(), BidirectionalGRUANDGRUModel(), 			RNNAttentionBidirectionalLSTMModel(), RNNAttentionBidirectionalGRUModel(), RNNAttentionLSTMModel(), 				RNNAttentionGRUModel(), RNNAttentionBidirectionalLSTMANDLSTMModel(), RNNAttentionBidirectionalGRUANDLSTMModel(), 				RNNAttentionBidirectionalLSTMANDGRUModel(), RNNAttentionBidirectionalGRUANDLSTMModel(),  				CNNBidirectionalLSTMModel(), CNNBidirectionalGRUModel(), CNNLSTMModel(), CNNGRUModel(), 				CNNBidirectionalLSTMANDLSTMModel(), CNNBidirectionalGRUANDLSTMModel(), CNNBidirectionalLSTMANDGRUModel(), 				CNNBidirectionalGRUANDGRUModel(), CNNAttentionBidirectionalLSTMModel(), CNNAttentionBidirectionalGRUModel(), 				CNNAttentionLSTMModel(), CNNAttentionGRUModel(), CNNAttentionBidirectionalLSTMANDLSTMModel(), 			CNNAttentionBidirectionalGRUANDLSTMModel(), CNNAttentionBidirectionalLSTMANDGRUModel(), 				CNNAttentionBidirectionalGRUANDGRUModel()
		]
		
	
