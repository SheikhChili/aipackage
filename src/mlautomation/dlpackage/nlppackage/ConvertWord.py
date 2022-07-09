#CONVERT WORDS TO INT PACKAGE
from aipackage.dlpackage.PackageVariable import Variable


class HandleWord():
	
	#init
	def __init__(self):
		self.dataUniqueWordsArray = []
		self.dataWordToInt = {}
		self.dataIntToWord = {}
		
		self.labelUniqueWordsArray = []
		self.labelWordToInt = {}
		self.labelIntToWord = {}
		
		self.trainSentenceInt = []
		self.testSentenceInt = []
		self.labelSentenceInt = []
		
		
	
	#GETTER
	def getDataUniqueWords(self):
		return self.dataUniqueWordsArray
		
	def getTrainSentenceInt(self):
		return self.trainSentenceInt
		
	def getLabelSentenceInt(self):
		return self.labelSentenceInt	
		
	def getTestSentenceInt(self):
		return self.testSentenceInt
			
	def getDataWordToInt(self):
		return self.dataWordToInt
		
	def getLabelUniqueWords(self):
		return self.labelUniqueWordsArray	
		
	def getLabelWordToInt(self):
		return self.labelWordToInt
		
	
	#SETTER
	def setDataWordToInt(self,wordToInt):
		self.dataWordToInt = wordToInt	
	
	def setDataUniqueWordsArray(self,array):
		self.dataUniqueWordsArray = array	
		
	def setLabelWordToInt(self,wordToInt):
		self.labelWordToInt = wordToInt	
	
	def setLabelUniqueWordsArray(self,array):
		self.labelUniqueWordsArray = array	
		
				
	def pad(self,l, content, width):
		l.extend([content] * (width - len(l)))
		return l
	    
	    
	def checkAndStoreUniqueWord(self,word,data_type):
		if data_type == Variable.TYPE_EXTRAS_LABEL:
			if word not in self.labelUniqueWordsArray:
				self.labelUniqueWordsArray.append(word)
				self.createLabelWordToIntDictionary(word, len(self.labelUniqueWordsArray))
		else:
			if word not in self.dataUniqueWordsArray:
				self.dataUniqueWordsArray.append(word)
				self.createTrainTestWordToIntDictionary(word, len(self.dataUniqueWordsArray))	


	def createTrainTestWordToIntDictionary(self,word, index):
		self.dataIntToWord[index] = word
		self.dataWordToInt[word] = index
		
	def createLabelWordToIntDictionary(self,word, index):
		self.labelIntToWord[index] = word
		self.labelWordToInt[word] = index


	def convertWordToInt(self, sentence_array, word_to_int, max_length, data_type):
		print("called word to int",data_type)
		for sentence in sentence_array:
			word_array = self.convertAndGetWordToInt(sentence, word_to_int, max_length)	
			if data_type == Variable.TYPE_EXTRAS_TRAIN:
				self.trainSentenceInt.append(word_array)
			elif data_type ==  Variable.TYPE_EXTRAS_TEST:
				self.testSentenceInt.append(word_array)	
			else:	
				self.labelSentenceInt.append(word_array)					
		print("finished word to int",data_type,"\n")
		
	def convertAndGetWordToInt(self, sentence, word_to_int, max_length):	
		word_array = []
		for word in sentence:
			word_array.append(word_to_int[word])
		self.pad(word_array, 0, max_length)
		return word_array
			
	def clearWordIntVariable(self):
		self.dataWordToInt = None
		self.dataIntToWord = None
		
		self.labelWordToInt = None
		self.labelIntToWord = None
		
		
	def clearTrainVariable(self):
		self.dataUniqueWordsArray = None
		self.trainSentenceInt = None
		
		self.labelUniqueWordsArray = None
		self.labelSentenceInt = None
		
	def clearTestVariable(self):
		self.testSentenceInt = []		
