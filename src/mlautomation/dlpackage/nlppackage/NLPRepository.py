

#import
import threading
import time
import os
import tensorflow as tf
import numpy as np
from aipackage.dlpackage.nlppackage.ConvertWord import HandleWord
from aipackage.dlpackage.StoreData import InputOutputStream
from aipackage.dlpackage.nlppackage.PreprocessText import PreprocessText
from aipackage.dlpackage.HyperParameterTuning import HyperParameterTuning
from aipackage.dlpackage.nlppackage.Entities import EmbeddingEntity
from aipackage.dlpackage.Entities import HyperParameterEntity
from aipackage.dlpackage.Repository import Repository
from aipackage.dlpackage.PackageVariable import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler


class NLPRepository(Repository):
	
	
	#init
	def __init__(self):
		super().__init__()
		self.updateDatasetLocation(Variable.nlpDatasetLocation)
		self.trainLabel = []
		
		self.textType = None
		
		#initialize	
		self.oriPreprocessTrainSentenceArray = []
		self.oriPreprocessTestSentenceArray = []
		self.oriLabelArray = []
		
		#class
		self.preprocessText = PreprocessText()
		self.handleWord = HandleWord()


	def setTrainLabel(self, labelArray):
		self.trainLabel = labelArray
		
	def setTextType(self, typeValue):
		self.textType = typeValue
		
		
	def getOriTrainData(self):
		return self.readPreprocessData(Variable.fileNamePrefix + Variable.trainPreprocessFileName)
			
			
	def setTrainLabelData(self):
		self.setOriPreprocessTrainSentenceArray(self.getOriTrainData())
		self.setOriLabelArray(self.readPreprocessData(Variable.fileNamePrefix + Variable.labelFileName))
		
		
	def setCharTrainLabelData(self, fileNamePrefix):
		self.setOriPreprocessTrainSentenceArray(self.readPreprocessData(fileNamePrefix + Variable.trainCharFileName))
		self.setOriLabelArray(self.readPreprocessData(fileNamePrefix + Variable.labelCharFileName))	
			
			
	def setAllTestData(self):
		fileNamePrefix = Variable.fileNamePrefix
		self.setOriPreprocessTestSentenceArray(self.readPreprocessData(fileNamePrefix + Variable.testPreprocessFileName))
		self.setAllConvertData()
		
		
	def setAllConvertData(self):
		fileNamePrefix = Variable.fileNamePrefix
		self.setWordConvertData(fileNamePrefix)
		
		
	def setWordConvertData(self, fileNamePrefix)	:
		self.setDataWordToInt(self.readDictData(fileNamePrefix + Variable.dataWordToIntFileName))
		self.setDataUniqueWordsArray(self.readPreprocessData(fileNamePrefix + Variable.dataUniqueWordFileName))	
		if(self.type == Variable.typeEncoderDecoder):
			self.setLabelWordToInt(self.readDictData(fileNamePrefix + Variable.labelWordToIntFileName))
			self.setLabelUniqueWordsArray(self.readPreprocessData(fileNamePrefix + Variable.labelUniqueWordFileName))
			
			
	def setCharConvertData(self, fileNamePrefix):	
		self.setDataWordToInt(self.readDictData(fileNamePrefix + Variable.charToIntFileName))
		self.setDataUniqueWordsArray(self.readPreprocessData(fileNamePrefix + Variable.uniqueCharFileName))	
		
			
	#getter	
	def getOriPreprocessTrainSentenceArray(self):
		return self.oriPreprocessTrainSentenceArray    
		
		
	def getOriPreprocessTestSentenceArray(self):
		return self.oriPreprocessTestSentenceArray 
		
		
	def getOriLabelArray(self):
		return self.oriLabelArray 	  
		
			
	def getDataUniqueWordsArray(self):
	   	return self.handleWord.getDataUniqueWords()
	   	
	   	
	def getDataWordToInt(self):
	   	return self.handleWord.getDataWordToInt()
	   	
	   	
	def getLabelUniqueWordsArray(self):
	   	return self.handleWord.getLabelUniqueWords()
	   	
	   	
	def getLabelWordToInt(self):
	   	return self.handleWord.getLabelWordToInt() 
	   	   
	   	   	   	
	def getTrainSentenceInt(self):
		return self.handleWord.getTrainSentenceInt()
		
		
	def getTestSentenceInt(self):
		return self.handleWord.getTestSentenceInt()
		
		
	def getLabelSentenceInt(self):
		return self.handleWord.getLabelSentenceInt()			   	
	   	
	   	
	def getCombinePreprocessSentence(self):
		return self.getOriPreprocessTrainSentenceArray() + self.getOriPreprocessTestSentenceArray() 
		   
		   	
	def getDataMaxSentenceLength(self):
		return len(max(self.getCombinePreprocessSentence(), key=len))	
	
	
	def getDataVocabSize(self):
		return len(self.getDataUniqueWordsArray()) + 1 
		
		
	def getLabelMaxSentenceLength(self):
		return len(max(self.getOriLabelArray(), key=len))	
	
	
	def getLabelVocabSize(self):
		return len(self.getLabelUniqueWordsArray()) + 1 	
		
		
	def getHandleWord(self):
		return self.handleWord
		
		
	def getTestSet(self):
		if self.getOriPreprocessTestSentenceArray() == []:
			if self.getOriPreprocessTrainSentenceArray() == []:
				self.setAllTrainData()
				return self.getOriPreprocessTrainSentenceArray()
			else:
				return self.getOriPreprocessTrainSentenceArray()
		else:
			return self.getOriPreprocessTestSentenceArray()
		
			
	#SETTER
	def setOriPreprocessTrainSentenceArray(self, array):
		self.oriPreprocessTrainSentenceArray = array
		
		
	def setOriPreprocessTestSentenceArray(self, array):
		self.oriPreprocessTestSentenceArray = array
		
		
	def setOriLabelArray(self, array):
		self.oriLabelArray = array
		
			
	def setDataWordToInt(self,wordToInt):
		self.handleWord.setDataWordToInt(wordToInt)	
		
		
	def setDataUniqueWordsArray(self,array):
		self.handleWord.setDataUniqueWordsArray(array)
		
		
	def setLabelWordToInt(self,wordToInt):
		self.handleWord.setLabelWordToInt(wordToInt)	
		
		
	def setLabelUniqueWordsArray(self,array):
		self.handleWord.setLabelUniqueWordsArray(array)	
			
			
	def normalizeData(self, data):
		min_max_scaler = StandardScaler()
		return min_max_scaler.fit_transform(data)
			
			
	def getXYData(self):
		X = self.getTrainSentenceInt()

		if(self.type == Variable.typeClassification):
			Y = super().get_ori_label_array()[:super().get_data_limit(Variable.TYPE_EXTRAS_LABEL)]
		else:	
			Y = self.getLabelSentenceInt()
		return super().get_xy_data(X, Y)
		
			
	def updatePreprocessData(self):
		stopWords = self.readDictData(Variable.stopWordFileLocation)
		contractionMapping = self.readDictData(Variable.contractionFileLocation)
		self.preprocessText.setPreprocessData(stopWords, contractionMapping)
		
		
			
	#BUSSINESS LOGIC METHODS
	def convertSentence(self, data_type):
		max_length = []
		word_to_int = []
		sentence_array = []
		if data_type == Variable.TYPE_EXTRAS_LABEL: 
			word_to_int = self.getLabelWordToInt() 
			max_length = self.getLabelMaxSentenceLength()
			sentence_array = self.getOriLabelArray()
		else: 
			word_to_int = self.getDataWordToInt()
			max_length = self.getDataMaxSentenceLength()
			if data_type == Variable.TYPE_EXTRAS_TRAIN:
				sentence_array = self.getOriPreprocessTrainSentenceArray()	
			else:
				sentence_array = self.getTestSet()	
		self.handleWord.convertWordToInt(sentence_array[:self.getDataLimit(data_type)], word_to_int, max_length, data_type)
		
		
	def getDataLimit(self, data_type):
		if data_type == Variable.TYPE_EXTRAS_TEST:
			return self.getTestDataLimit()
		else:
			return self.getTrainDataLimit()
			
					
	def getTrainDataLimit(self):
		trainDataLimit = Variable.trainDataLimit
		if self.wantTrainAllData:
			trainDataLimit = len(self.getOriPreprocessTrainSentenceArray())
		return trainDataLimit
		
		
	def getTestDataLimit(self):
		testDataLimit = Variable.testDataLimit
		if self.wantTestAllData:
			testDataLimit = len(self.getOriPreprocessTestSentenceArray())
		return testDataLimit			  	


	def storeDataInArray(self,data,data_type, index=None):
		if data_type == Variable.TYPE_EXTRAS_TRAIN:
			self.oriPreprocessTrainSentenceArray.append(data)
			data = None
			labelData = self.trainLabel[index]
			if(self.type == Variable.typeClassification):
				data = labelData
			else:
				if self.textType == Variable.TYPE_WORD:
					data = self.modifyWord(labelData, Variable.TYPE_EXTRAS_LABEL)
				else:
					data = self.modifyCharData(labelData, Variable.TYPE_EXTRAS_LABEL)	
			self.storeDataInArray(data, Variable.TYPE_EXTRAS_LABEL)		
		elif data_type == Variable.TYPE_EXTRAS_LABEL:
			self.oriLabelArray.append(data)	
		else:	
			self.oriPreprocessTestSentenceArray.append(data)
			
	  	
	def modifyText(self,sentence_array, data_type, num=0):
		i = 0
		for sentence in sentence_array:
			index = ((num*500) + i)
			text_array = []
			if self.textType == Variable.TYPE_WORD:
				text_array = self.modifyWord(sentence, data_type)
			else:
				text_array = self.modifyCharData(sentence, data_type)	
			#if(len(text_array) < 1000) or data_type == Variable.TYPE_EXTRAS_TEST:
			self.storeDataInArray(text_array, data_type, index)
			i+=1    
		print('finished',data_type)
	  	
	  	
	def modifyWord(self, sentence, data_type):
		word_array = []
		for word in self.preprocessText.processWordSentence(sentence):
			if str(word) not in self.preprocessText.getStopWords():
				word = self.preprocessText.processWord(word)
				if(word != None):
					self.handleWord.checkAndStoreUniqueWord(word, data_type)
					word_array.append(word)
		return word_array
		
		
	def modifyCharData(self,data, data_type):
		data = self.preprocessText.processCharSentence(data)
		char_array = []
		if data_type == Variable.TYPE_EXTRAS_LABEL:
			char_array = data
		else:
			char_array = list(data)	
		uniqueChars = set(char_array)
		for char in uniqueChars:
			self.handleWord.checkAndStoreUniqueWord(char, data_type)
		return char_array				
					
						 	
	def storeData(self, fileNamePrefix, ioStream):
	
		super().store_data(fileNamePrefix, ioStream)
		t1 = threading.Thread(target=ioStream.store_preprocess_array, args=(fileNamePrefix + Variable.dataUniqueWordFileName, self.getDataUniqueWordsArray()))
		t2 = threading.Thread(target=ioStream.store_dict, args=(fileNamePrefix + Variable.dataWordToIntFileName, self.getDataWordToInt()))

		# start array storing thread
		t1.start()
		t2.start()
		#join
		t1.join()
		t2.join()
		
		
	def processTrainSentence(self,ori_train_sentences):
		j = 0
		for i in range(0, len(ori_train_sentences) + 500, 500):
			t1 = threading.Thread(target=self.modifyText, args=(ori_train_sentences[i:i + 500], Variable.TYPE_EXTRAS_TRAIN, j))
			t1.start()
			t1.join()
			j+=1
			
				
	def processTestSentence(self,ori_test_sentences):	
		k = 0
		for i in range(0, len(ori_test_sentences) + 500, 500):
		    k += 1
		    t1 = threading.Thread(target=self.modifyText, args=(ori_test_sentences[i:i + 500], Variable.TYPE_EXTRAS_TEST, k))
		    t1.start()
		    t1.join()
		 
		    
	def readDictData(self, filename):
		return InputOutputStream().read_dict(filename)
		
	def readPreprocessData(self, filename):
		return InputOutputStream().get_preprocess_array(filename)
		
		
	def getMaxScoreAndValScore(self):
		io = InputOutputStream()
		df = io.get_stored_model_data(Variable.modelFileName)
		if(df.empty):
			score,val_score = 0,0
		else:
			val_score = max(df[Variable.valScoreName])
			rslt_df = df[df[Variable.valScoreName] == val_score]
			score = max(rslt_df[Variable.scoreName])
		return score, val_score
		
		
	def getStoredModelFileNameArray(self, fileName = Variable.modelFileName):
		io = InputOutputStream()
		return io.get_stored_model_file_name_array(fileName)
			
			
	def updateStoredModelFileNameArray(self):
		self.storedModelFileNameArray = self.getStoredModelFileNameArray()
		
		
	def getCount(self):
		if(super().get_type()):
			return Variable.nlp_layer_count
		else:
			return Variable.enc_dec_layer_count	
		
			
	def runAllModels(self, modelList, modelEntity, datasetEntity):
		print("RUNALLMODELS\n")
		
		dataVocabSize = self.getDataVocabSize()
		print("Vocab size = ", dataVocabSize)
		dataMaxLen = self.getDataMaxSentenceLength()
		print("MAXIMUM LENGTH SENTENCE = ", dataMaxLen,"\n")
		
		
		#self.clearAfterTrainStartVariable()
		hyperParameterTuning = HyperParameterTuning(datasetEntity)	
		scores = super().get_max_score_and_val_score()
		hyperParameterTuning.set_scores(scores[0], scores[1])
		hyperParameterTuning.set_type(self.getType())
		super().update_stored_model_file_name_array()
		for embedding_dim in Variable.embedding_dim_array:
			#embedding_matrix = self.getEmbeddingMatrix(dataVocabSize, embedding_dim)
			for i in range(0,len(modelList)):
				model = modelList[i]
				modelEntity.modelName = model.__class__.__name__.replace("Model","")
				model.set_model_data(modelEntity)
				hyperParameterEntity = HyperParameterEntity(model, i)
			
				#self.runModelWithEmbeddingMatrix(embedding_dim, dataMaxLen, embedding_matrix, hyperParameterEntity, hyperParameterTuning)
				self.runModelWithoutEmbeddingMatrix(embedding_dim, dataMaxLen, dataVocabSize, hyperParameterEntity, hyperParameterTuning)
			
	def runModelWithEmbeddingMatrix(self, embedding_dim, dataMaxLen, embedding_matrix, hyperParameterEntity, hyperParameterTuning):
		if not super().check_file_exist(self.getEmbeddingFileLocation(embedding_dim)):
			return
		model = hyperParameterEntity.model	
		model.set_embedding_data(EmbeddingEntity(dataMaxLen, embedding_matrix = embedding_matrix))
		hyperParameterEntity.modelClassName = model.__class__.__name__ + Variable.typeEmbedding + str(embedding_dim) + str(super().getCount())
		self.runModel(hyperParameterEntity, hyperParameterTuning)
		
		
	def runModelWithoutEmbeddingMatrix(self, embedding_dim, dataMaxLen, dataVocabSize, hyperParameterEntity, hyperParameterTuning):
		model = hyperParameterEntity.model
		model.set_embedding_data(EmbeddingEntity(dataMaxLen, dataVocabSize, embedding_dim))
		hyperParameterEntity.modelClassName = model.__class__.__name__ + Variable.filenameSeparator + str(embedding_dim)
		self.runModel(hyperParameterEntity, hyperParameterTuning)


	def runModel(self, hyperParameterEntity, hyperParameterTuning):
		super().run_model(hyperParameterEntity, hyperParameterTuning)


	def clearPreprocessVariable(self):
		print("clearPreprocessVariable called-------\n")
		self.preprocessText.clearAllGlobalVariable()	
		
		
	def clearAfterConvertWordNumericVariable(self):
		print("clearAfterConvertWordNumericVariable called-------\n")
		self.handleWord.clearWordIntVariable()
		self.trainLabel = None
		
		
	def clearAfterTrainStartVariable(self):
		print("clearAfterTrainStartVariable called-------\n")	
		self.handleWord.clearTrainVariable()
		self.oriPreprocessTrainSentenceArray = None
		self.oriLabelArray = None
		
		
	def clearAfterTestVariable(self):
		print("clearAfterTestVariable called-------\n")
		self.handleWord.clearTestVariable()
		self.oriPreprocessTestSentenceArray = []
			
			
	def convertTestSentence(self):
		t1 = threading.Thread(target=self.convertSentence, args=([Variable.TYPE_EXTRAS_TEST]))
		t1.start()
		t1.join()
		
		
	def processAndSetTestData(self):
		self.setAllTestData()
		if(self.getOriPreprocessTestSentenceArray() == []):
			if(self.getOriPreprocessTrainSentenceArray() == []):
				self.setOriPreprocessTestSentenceArray(self.getOriTrainData())
			else:
				self.setOriPreprocessTestSentenceArray(self.getOriPreprocessTrainSentenceArray()[:self.getTestDataLimit()])

		
    					
	def startPredict(self, subEntity):
		print("PREDICT STARTS \n")
		self.processAndSetTestData()
		#if not self.isModelPresent():
		#	return 
		self.convertAndPredict(subEntity)
		
		
	def convertAndPredict(self, subEntity = None):
		super().setTestData(np.array(self.getTestSentenceInt()))
		super().start_predict(self, sub_entity= None, predict_label_dict= None)
			
						
	def startNewPredict(self, sentence):
		print("\nNEW PREDICT START")
		self.setAllConvertData()
		print()
		self.clearAfterTestVariable()		
		self.modifyText([sentence], Variable.TYPE_EXTRAS_TEST)
		print()		
		self.convertAndPredict()
		
		
	'''def runGradio(self,func = self.startNewPredict):
		iface = gr.Interface(fn=func, inputs="text",outputs="text")
		iface.launch()'''		
		
			
	def getEmbeddingIndex(self, embedding_dim):
		embeddings_index = {}
		embeddingFileLocation = self.getEmbeddingFileLocation(embedding_dim)
		with open(embeddingFileLocation) as f:
			for line in f:
				word, coefs = line.split(maxsplit=1)
				coefs = np.fromstring(coefs, "f", sep=" ")
				embeddings_index[word.lower()] = coefs

		return embeddings_index
		
		
	def getEmbeddingMatrix(self, dataVocabSize, embedding_dim):
		embeddings_index = self.getEmbeddingIndex(embedding_dim)
		word_to_int = self.getDataWordToInt()
		
		hits = 0
		misses = 0

		# Prepare embedding matrix
		embedding_matrix = np.zeros((dataVocabSize, embedding_dim))
		for word, i in word_to_int.items():
			
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# Words not found in embedding index will be all-zeros.
				# This includes the representation for "padding" and "OOV"
				embedding_matrix[i] = embedding_vector
				hits += 1
			else:
				misses += 1
		print("Converted %d words (%d misses)" % (hits, misses),"\n\n")
		return embedding_matrix
	
	
	def getEmbeddingFileLocation(self, embedding_dim):
		return Variable.embeddingFileLocationPrefix + Variable.embeddingFileNamePrefix + str(embedding_dim) +  Variable.embeddingFileNameSuffix
