

#import
import threading
import numpy as np
from aipackage.dlpackage.StoreData import InputOutputStream
from aipackage.dlpackage.nlppackage.nlpmanytoonepackage.NLPManyToOneModel import NLPManyToOneModel
from aipackage.dlpackage.nlppackage.NLPRepository import NLPRepository
from aipackage.dlpackage.PackageVariable import Variable




class NLPManyToOneRepository(NLPRepository):
		
	def __init__(self):
		super().__init__()
		
	def preprocess(self, ori_train_sentences, ori_test_sentences):
		super().setTextType(Variable.TYPE_WORD)
		super().updatePreprocessData()
		self.checkOrCreateRequiredDataExist(ori_train_sentences, ori_test_sentences)
		#super().clearPreprocessVariable()
			
	def checkOrCreateRequiredDataExist(self, ori_train_sentences, ori_test_sentences=[]):
		if(super().is_data_present()):
			super().set_all_train_data()
		else:
			self.preprocessData(ori_train_sentences, ori_test_sentences)
			self.checkAndStoreData()			
			
	def preprocessData(self, ori_train_sentences, ori_test_sentences = []):		
		super().processTrainSentence(ori_train_sentences)
		if(ori_test_sentences != []):
			super().processTestSentence(ori_test_sentences)
	
	def checkAndStoreData(self):
		if(super().get_ori_preprocess_train_sentence_array()==[]):
			return
		fileNamePrefix = Variable.preprocessFolderName + Variable.locationSeparator
		ioStream = InputOutputStream()	
		super().store_data(fileNamePrefix, ioStream)
		t1 = threading.Thread(target=ioStream.store_preprocess_array, args=(fileNamePrefix + Variable.labelFileName, super().get_ori_label_array()))
		t1.start()
		t1.join()	
		
	
	def updateUserMetrics(self, userMetrics = None):
		super().update_user_metrics(userMetrics)
		
			
	def convertTrainSentence(self):
		t1 = threading.Thread(target=super().convertSentence, args=([Variable.TYPE_EXTRAS_TRAIN]))
		t1.start()
		t1.join()
			
	def createAndRunModel(self, entities):	
		nlpClassifyModel = NLPManyToOneModel()
		#super().clearAfterConvertWordNumericVariable()
		super().run_all_models(nlpClassifyModel.getAllNLPManyToOneModels(), entities[0], entities[1])
