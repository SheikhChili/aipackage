

#import
import numpy as np
import pandas as pd
import re
import threading
from aipackage.nlppackage.StoreData import InputOutputStream
from aipackage.nlppackage.PreprocessText import PreprocessText
from aipackage.nlppackage.PackageVariable import Variable
from aipackage.nlppackage.ConvertWord import HandleWord

class ScriptDatasetRepository():

	def __init__(self):
	
		#CLASS
		self.preprocessText = PreprocessText()	
		self.handleWord = HandleWord()
		self.contraction_mapping = {
					"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": 						"could have","couldn't": "could not","didn't": "did not", "doesn't": "does not", "don't": "do 					not", "hadn't": "had not","hasn't": "has not", "haven't": "have not","he'd": "he would","he'll": 						"he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 					"how's": "how is","I'd": "I would", "I'd've": "I would have","I'll": "I will", "I'll've": "I 						will have","I'm": "I am", "I've": "I have", "i'd": "i would","i'd've": "i would have", "i'll": 					"i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": 						"it would","it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": 					"it is", "let's": "let us", "ma'am": "madam","mayn't": "may not", "might've": "might 						have","mightn't": "might not","mightn't've": "might not have", "must've": "must have","mustn't": 						"must not", "mustn't've": "must not have", "needn't": "need not","needn't've": "need not 						have","o'clock": "of the clock","oughtn't": "ought not", "oughtn't've": "ought not have", 						"shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have","she'd": "she 						would", "she'd've":"she would have", "she'll": "she will", "she'll've": "she will have", 						"she's": "she is",     "should've": "should have", "shouldn't": "should not", "shouldn't've": 					"should not have", "so've": "so have","so's": "so as","this's": "this is","that'd": "that 						would", "that'd've": "that would have", "that's": "that is", "there'd": "there 					would","there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": 					"they would","they'd've": "they would have","they'll": "they will", "they'll've": "they will 						have", "they're": "they are", "they've": "they have", "to've": "to have","wasn't": "was not", 					"we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 						"we're": "we are","we've": "we have", "weren't": "were not", "what'll": "what will", 						"what'll've": "what will have", "what're": "what are","what's": "what is", "what've":"what 						have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where 					is","where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who 						is", "who've":"who have","why's": "why is", "why've": "why have", "will've": "will have", 						"won't": "will not", "won't've": "will not have","would've": "would have", "wouldn't": "would 					not", "wouldn't've": "would not have", "y'all": "you all","y'all'd": "you all 						would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all 						have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": 						"you will have","you're": "you are", "you've": "you have", "im": "i am"
		}
		self.oriPreprocessTrainSentenceArray = []
		self.oriLabelArray = []
		
	#getter	
	def getOriPreprocessTrainSentenceArray(self):
		return self.oriPreprocessTrainSentenceArray    
		
	def getOriLabelArray(self):
		return self.oriLabelArray 	
			
	def getDataUniqueWordsArray(self):
		return self.handleWord.getDataUniqueWords()
		
	def getDataWordToInt(self):
		return self.handleWord.getDataWordToInt()
				
	def createDataset(self, train, max_len, gap_char = ''):
		self.oriPreprocessTrainSentenceArray = []
		self.oriLabelArray = []
		n_chars = len(train)
		word_to_int = self.getDataWordToInt()
		for i in range(0, n_chars - max_len, 1):	
			seq_in = train[i:i + max_len]			
			seq_out = word_to_int[train[i + max_len]]
			#seq_in = gap_char.join(seq_in)	
			self.oriPreprocessTrainSentenceArray.append(seq_in)	
			self.oriLabelArray.append(seq_out)	
		
	# changing some notations
	def clean_text(self,text):
		text = text.lower()
		text = ' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in text.split(" ")])
		#repacing 'nt,'d checking the dictionary and replacing it
		text = text.replace("\n", " \n ")
		text = text.replace("\r", " \r ")
		text = text.replace(" u ", " you ")
		text = text.replace(" v ", " we ")
		text = text.replace(" & ", " and ")
		text = text.replace("'s", " is ")

		# Numbers with word
		text = text.replace("0", " zero ")
		text = text.replace("1", " one ")
		text = text.replace("2", " two ")
		text = text.replace("3", " three ")
		text = text.replace("4", " four ")
		text = text.replace("5", " five ")
		text = text.replace("6", " six ")
		text = text.replace("7", " seven ")
		text = text.replace("8", " eight ")
		text = text.replace("9", " nine ")

		# remove links
		text = re.sub(r'https?://\S+', '', text, flags=re.MULTILINE)
		text = re.sub(r'http?://\S+', '', text, flags=re.MULTILINE)

		# remove punctuation
		text = re.sub(r'([!@#$%?\-_=\^&\*\(\):;"\',\./\\]+)', r' \1 ', text)
		return text
		
	# preprocessing
	def processWordSentence(self,sentence):
		sentence = self.preprocessText.encodeSentence(sentence)
		sentence = self.preprocessText.strip_html_tags(sentence)
		sentence = self.clean_text(sentence)
		sentence = self.preprocessText.getNLPSentence(sentence)
		return sentence
		
	def preprocessWord(self,sentence):
		word_array = []
		for word in self.processWordSentence(sentence):
			word = self.processWord(word)
			self.handleWord.checkAndStoreUniqueWord(word, Variable.TYPE_EXTRAS_TRAIN)
			word_array.append(word)
		return word_array
		
	def processWord(self,word):		
		if word.lemma_ != "-PRON-":
			# lemmas
			word = word.lemma_
			# spell correction(this make the code to run slow)
			# word = str(TextBlob(word).correct())
			return word
		return word.orth_
		
	def processAndStoreWord(self, ori_train_sentences, folderName):
		fileNamePrefix = Variable.scriptLocation + folderName + Variable.locationSeparator
		
		data = self.preprocessWord(ori_train_sentences)
		
		train = self.createDataset(data,100," ")

		self.storeConvertData(fileNamePrefix)
		
		#self.storeData(train, fileNamePrefix + Variable.trainWordFileName)
		
	def createDir(self, folderName):
		filenamePrefix = Variable.scriptLocation + folderName
		io = InputOutputStream()
		io.check_and_create_dir(filenamePrefix)
		
	def processAndStoreChar(self, ori_train_sentences, folderName):
		fileNamePrefix = Variable.scriptLocation + folderName + Variable.locationSeparator
		
		data = self.processCharSentence(ori_train_sentences)
		data = list(data)
		
		self.setCharData()
		
		self.createDataset(data,100)
		
		self.storeTrainLabelData(fileNamePrefix, Variable.TYPE_CHAR)
		#self.storeData(train, fileNamePrefix + Variable.trainCharFileName)		
		
	def storeData(self, train, filename):		
		df = pd.DataFrame(train, columns=["sentence","label"])
		df.to_csv(filename, index = False)
		
	# preprocessing
	def processCharSentence(self,sentence):
		sentence = self.preprocessText.encodeSentence(sentence)
		sentence = sentence.lower()
		return sentence	
		
	def storeConvertData(self, fileNamePrefix):
		self.storeTrainLabelData(fileNamePrefix)
		ioStream = InputOutputStream()
		t1 = threading.Thread(target=ioStream.store_data, args=(fileNamePrefix + Variable.dataUniqueWordFileName,  self.getDataUniqueWordsArray()))
		t2 = threading.Thread(target=ioStream.store_data, args=(fileNamePrefix + Variable.dataWordToIntFileName,   self.getDataWordToInt()))
		
		t1.start()
		t2.start()
		
		t1.join()
		t2.join()

		
	def storeTrainLabelData(self, fileNamePrefix, textType = Variable.TYPE_WORD):
		ioStream = InputOutputStream()
		trainFileName = fileNamePrefix + Variable.trainPreprocessFileName
		labelFileName = fileNamePrefix + Variable.labelFileName
		if textType == Variable.TYPE_CHAR:
			trainFileName = fileNamePrefix + Variable.trainCharFileName
			labelFileName = fileNamePrefix + Variable.labelCharFileName
		
		t1 = threading.Thread(target=ioStream.store_data, args=(trainFileName,  self.getOriPreprocessTrainSentenceArray()))
		t2 = threading.Thread(target=ioStream.store_data, args=(labelFileName,  self.getOriLabelArray()))
		
		t1.start()
		t2.start()
		
		t1.join()
		t2.join()
		
		
	'''def createCharInt(self):
		ioStream = InputOutputStream()
		uniqueChar = ioStream.read_data("uniqueChar.txt")
		char_int = {}
		for index,char in enumerate(uniqueChar):
			char_int[char] = index + 1
		ioStream.store_data("charToInt.txt",char_int)'''
		
	def setCharData(self):
		ioStream = InputOutputStream()	
		fileNamePrefix = Variable.charDataLocation
		self.handleWord.setDataWordToInt(ioStream.read_data(fileNamePrefix + Variable.charToIntFileName))
		self.handleWord.setDataUniqueWordsArray(ioStream.read_data(fileNamePrefix + Variable.uniqueCharFileName))			
		
		
			   		
