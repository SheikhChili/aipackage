

#import
import threading
import numpy as np
from random import randint
from aipackage.nlppackage.StoreData import InputOutputStream
from aipackage.nlpclassification.NLPClassifyRepository import NLPClassifyRepository
from aipackage.nlppackage.PackageVariable import Variable



class TGRepository(NLPClassifyRepository):
	
	def __init__(self, folderName):
		self.folderName = folderName
		super().__init__()
		
	def preprocessWord(self):
		super().setTextType(Variable.TYPE_WORD)
		preprocessFilenamePrefix = Variable.textGenerateDatasetLocation + Variable.scriptLocation + self.folderName + Variable.locationSeparator 
		super().set_train_label_data(preprocessFilenamePrefix)
		super().setWordConvertData(preprocessFilenamePrefix)
		
	def preprocessChar(self):
		super().setTextType(Variable.TYPE_CHAR)
				
		preprocessFilenamePrefix = Variable.textGenerateDatasetLocation + Variable.scriptLocation + self.folderName + Variable.locationSeparator 
		super().setCharTrainLabelData(preprocessFilenamePrefix)
		preprocessFilenamePrefix = Variable.textGenerateDatasetLocation + Variable.charDataLocation
		super().setCharConvertData(preprocessFilenamePrefix)	
		
	
	def updateUserMetrics(self, userMetrics = None):
		super().update_user_metrics(userMetrics)
		
	
	def startPredict(self, sentenceArray):
		if not super().is_model_present():
				return
		print("SENTENCE ARRAY = ",sentenceArray,"\n")
		self.generateSentence(super().getTestSet()[:1])	
		print("\n")
			
	def generateSentence(self, sentenceArray):
		model = super().get_saved_model()
		model.summary()
		
		word_to_int = super().getDataWordToInt()
		max_len = super().getDataMaxSentenceLength()
		j = 0
		for i in range(Variable.textGenerationTestWordsCount):
			array = sentenceArray
			if (len(sentenceArray)>=max_len):
				array = sentenceArray[j:]
				j+=1
			print("ARRAY = ",array)
			token_list = super().getHandleWord().convertAndGetWordToInt(array, word_to_int, max_len)
			pred = model.predict_classes([token_list])
			word = list(word_to_int.keys())[list(word_to_int.values()).index(pred[0]+1)]
			print(word)
			sentenceArray.append(word)
			print(pred,"\n")
		print("\n\n\n"," ".join(sentenceArray))	
		
	def askNewSentence(self):
		condition = input("If you want to try your own sentence Y/N = ")
		yesArray = ["Y", "y", "yes", "YES"]
		if condition in yesArray:
			sentence = input("Enter your sentence to predict = ")
			super().set_ori_preprocess_test_sentence_array([])
			sentenceArray = super().modifyText([sentence], Variable.TYPE_EXTRAS_TEST)
			self.startPredict(super().get_ori_preprocess_test_sentence_array()[0])
