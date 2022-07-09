

#import
import numpy as np
from aipackage.dlpackage.StoreData import InputOutputStream
from aipackage.dlpackage.nlppackage.nlpmanytoonepackage.NLPManyToOneRepository import NLPManyToOneRepository
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.Entities import DatasetEntity, ModelEntity




class NLPClassifyRepository(NLPManyToOneRepository):
		
	def __init__(self):
		super().__init__()
		super().set_type(Variable.typeClassification)
				
	def startTrain(self):
		super().convertTrainSentence()
		super().create_and_run_model(self.getEntity())
				
	def getEntity(self):
		array = super().get_xy_data()
		X = array[0]
		y = array[1]
		
		print("UNIQUE COUNTS = ",np.unique(y, return_counts=True), "\n\n")
		# encode class values as integers
		encoder = LabelEncoder()
		encoder.fit(y)
		encoded_Y = encoder.transform(y)
		
		# convert integers to dummy variables (i.e. one hot encoded)
		Y = to_categorical(encoded_Y)
		print("TRAIN SENTENCE SHAPE = ", X.shape)
		print("LABEL = ",Y.shape,"\n")
		
		return self.getModelEntity(Y.shape[1]), DatasetEntity(X,Y)			
	  	
	def getModelEntity(self, targetSize):
		final_activation = 'softmax'
		loss = 'categorical_crossentropy'
		optimizer = 'adam'
		metrics = super().get_metrics()
		return ModelEntity(targetSize, final_activation, loss, optimizer, metrics)
