

#import
from aipackage.nlppackage.StoreData import InputOutputStream
from aipackage.nlpmanytoonepackage.NLPManyToOneRepository import NLPManyToOneRepository
from aipackage.nlppackage.PackageVariable import Variable
from aipackage.nlppackage.Entities import DatasetEntity, ModelEntity




class NLPRegressionRepository(NLPManyToOneRepository):
		
	def __init__(self):
		super().__init__()
		super().set_type(Variable.typeRegression)
		
	def startTrain(self):
		super().convertTrainSentence()
		self.createAndRunModel(self.getEntity())
		
	def getEntity(self):
		array = super().get_xy_data()
		X = array[0]
		y = array[1]
		
		print("UNIQUE COUNTS = ",np.unique(y, return_counts=True), "\n\n")
		
		print("TRAIN SENTENCE SHAPE = ", X.shape)
		print("LABEL = ",Y.shape,"\n")
		
		return self.getModelEntity(Y.shape[1]), DatasetEntity(X,Y)			
	  	
	def getModelEntity(self, targetSize):
		final_activation = 'linear'
		loss = 'mse'
		optimizer = 'adam'
		metrics = super().get_metrics()
		return ModelEntity(targetSize, final_activation, loss, optimizer, metrics)
