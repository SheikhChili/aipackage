#PREPROCESS TEXT



#IMPORTS
import re
import sys
import spacy
import unidecode
from bs4 import BeautifulSoup
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.StoreData import InputOutputStream
'''try:
	from importlib import reload  # Python 3.4+
except ImportError:
	from imp import reload 

reload(sys)
#sys.setdefaultencoding('utf8')
PYTHONIOENCODING="UTF-8" ''' 


class PreprocessText():

	#init
	def __init__(self):
		self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
		# nlp = spacy.load('en_core_web_md')

		self.stopWords = None
		self.contraction_mapping = None
				     
	# changing some notations
	def clean_text(self,text):
		text = text.lower()
		text = ' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in text.split(" ")])
		#repacing 'nt,'d checking the dictionary and replacing it
		text = text.replace("\n", " ")
		text = text.replace("\r", " ")
		text = text.replace(" u ", " you ")
		text = text.replace(" v ", " we ")
		text = text.replace(" & ", " and ")

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
		text = re.sub(r'[^\w\s]', ' ', text)
		text = self.remove_whitespace(text)
		return text


	def remove_whitespace(self,text):
		"""remove extra whitespaces from text"""
		text = text.strip()
		return " ".join(text.split())


	def strip_html_tags(self,text):
		"""remove html tags from text"""
		soup = BeautifulSoup(text, "html.parser")
		stripped_text = soup.get_text(separator=" ")
		return stripped_text

	def getNLPSentence(self, sentence):
		return self.nlp(sentence)

	# preprocessing
	def processWordSentence(self,sentence):
		sentence = self.encodeSentence(sentence)
		sentence = self.strip_html_tags(sentence)
		sentence = self.clean_text(sentence)
		sentence = self.getNLPSentence(sentence)
		return sentence
		
	def processWord(self,word):		
		if word.lemma_ != "-PRON-":
			# lemmas
			word = word.lemma_
			# spell correction(this make the code to run slow)
			# word = str(TextBlob(word).correct())
			return word

	def encodeSentence(self, sentence):
		return unidecode.unidecode(str(sentence))
			
	def getStopWords(self):
		return self.stopWords
		
	def clearAllGlobalVariable(self):
		self.nlp = None
		self.stopWords = None
		self.contraction_mapping = None
		
	def setPreprocessData(self, stopWords, contractionMapping):
		self.stopWords = stopWords
		self.contraction_mapping = contractionMapping
				 	 	
