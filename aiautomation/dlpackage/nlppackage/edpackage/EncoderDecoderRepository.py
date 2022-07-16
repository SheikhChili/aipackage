

#import
import threading
import numpy as np
from aipackage.nlppackage.StoreData import InputOutputStream
from aipackage.edpackage.EncoderDecoderModel import EDModel
from aipackage.nlppackage.Repository import Repository
from aipackage.nlppackage.PackageVariable import Variable
from aipackage.nlppackage.Entities import DatasetEntity, ModelEntity
from aipackage.nlppackage.Models.ModelUtils import ModelUtils
from aipackage.nlppackage.Entities import AccuracyEntity
import warnings


class EDRepository(Repository):				

	def __init__(self):
		super().__init__()
		super().set_type(Variable.typeEncoderDecoder)
		
		
		
	def preprocess(self, ori_train_sentences, ori_test_sentences):
		super().setTextType(Variable.TYPE_WORD)
		super().updatePreprocessData()
		self.checkOrCreateRequiredDataExist(ori_train_sentences, ori_test_sentences)
			
	def checkOrCreateRequiredDataExist(self, ori_train_sentences, ori_test_sentences=[]):
		if(super().is_data_present()):
			super().set_all_train_data()
		else:
			self.preprocessData(ori_train_sentences, ori_test_sentences)
			self.checkAndStoreData()
			
	def preprocessData(self, ori_train_sentences, ori_test_sentences):		
		super().processTrainSentence(ori_train_sentences)
		if(ori_test_sentences != []):
			super().processTestSentence(ori_test_sentences)
				
	def checkAndStoreData(self):
		if(super().get_ori_preprocess_train_sentence_array()==[]):
			return
		fileNamePrefix = Variable.preprocessFolderName + Variable.locationSeparator	
		ioStream = InputOutputStream()		
		super().store_data(fileNamePrefix, ioStream)
		t1 = threading.Thread(target=ioStream.store_preprocess_array, args=(fileNamePrefix + Variable.labelUniqueWordFileName, self.getLabelUniqueWordsArray()))
		t2 = threading.Thread(target=ioStream.store_dict, args=(fileNamePrefix + Variable.labelWordToIntFileName, self.getLabelWordToInt()))
		t3 = threading.Thread(target=ioStream.store_preprocess_array, args=(fileNamePrefix + Variable.labelFileName, super().get_ori_label_array()))
		# start array storing thread
		t1.start()
		t2.start()
		t3.start()
		#join
		t1.join()
		t2.join()
		t3.join()
		
		
	def updateUserMetrics(self, userMetrics = None):
		super().update_user_metrics(userMetrics)
			
		
	def startTrain(self):
		self.convertTrainSentence()
		#super().clearAfterConvertWordNumericVariable()
		self.createAndRunModel()
			
			
	def convertTrainSentence(self):
		t1 = threading.Thread(target=super().convertSentence, args=([Variable.TYPE_EXTRAS_TRAIN]))
		t2 = threading.Thread(target=super().convertSentence, args=([Variable.TYPE_EXTRAS_LABEL]))
		t1.start()
		t2.start()
		t1.join()
		t2.join()
		
					
	def createAndRunModel(self):
		edModel = EDModel()
        super().run_all_models(edModel.getAllEDModels(), self.getModelEntity(), self.getDatasetEntity(), )
		
			
	def getDatasetEntity(self):
		x,y = super().get_xy_data()
		print("TRAIN SENTENCE SHAPE = ", x.shape)
		print("LABEL = ",y.shape,"\n")
		X = [x,y[:,:-1]]
		Y = y.reshape(y.shape[0],y.shape[1], 1)[:,1:]
		return DatasetEntity(X,Y)	
		
		
	def getModelEntity(self):
		labelMaxLen = super().getLabelMaxSentenceLength()
		print("MAXIMUM LABEL LENGTH SENTENCE = ", labelMaxLen)
		labelVocabSize = super().getLabelVocabSize()
		print("LABEL Vocab size = ", labelVocabSize,"\n")
		final_activation = 'softmax'
		loss = 'sparse_categorical_crossentropy'
		optimizer = 'adam'
		metrics = super().get_metrics()
		return ModelEntity(labelVocabSize, final_activation, loss, optimizer, metrics)
		
		
		
	def startPredict(self, subEntity):
		print("PREDICT STARTS \n")
		#if not self.isModelPresent():
		#	return 
		self.processAndSetTestData()
		self.convertAndPredict(subEntity)
		
			
	def convertAndPredict(self, subEntity = None):
		super().convertTestSentence()
		#self.clearAfterConvertWordNumericVariable()
		
		
		X_test = np.array(super().getTestSentenceInt())
		self.clearAfterTestVariable()
		
		#model = super().getBestModel(modelFileName)
		
		#pred = self.predict(X_test, subEntity, self.getInferenceModel(model), model.name)
		pred = self.predictAllModel(X_test, subEntity)	
		
		if(subEntity != None):
			subEntity.predictions = pred
			print("\n")
			super().submit_predictions(subEntity)
		else:
			return pred[0]	
		
				
	def predictAllModel(self, X_test, subEntity):
		storedPredictNameArray = super().get_stored_model_file_name_array(Variable.predictFileName)
		print(storedPredictNameArray)
		for modelFileName in super().get_all_saved_model_file_name():
			fullModelName = self.getFullModelName(modelFileName)
			if(fullModelName in storedPredictNameArray):
				continue
			
			print("\n\n" + fullModelName + " _ _  PREDICT THE DATA")
			
			model = super().get_loaded_model(modelFileName)
			
			self.predict(X_test, subEntity, self.getInferenceModel(model), model.name)
			super().save_predict_file_name(fullModelName)
		return []	
			
		
	def getDecLayerName(self, modelName):
		return modelName.split(Variable.decoder_name)[-1]
						
	def predict(self, X_test, subEntity, inferenceModel, modelName):
		print("\nTEST SET SIZE = ",len(X_test))
		encoder_model, decoder_model = inferenceModel
		print("\n\n ENCODER MODEL SUMMARY \n")
		encoder_model.summary()
		print("\n\n DECODER MODEL SUMMARY \n")
		decoder_model.summary()
		max_len_text = super().getDataMaxSentenceLength()
		pred = []
		for test in X_test:
			predictionText = self.decode_sequence(test.reshape(1,max_len_text), encoder_model, decoder_model, modelName)
			print("PREDICTION = ",predictionText)
			pred.append(predictionText)
		return pred
		
					
	def getInferenceModel(self, model):
		print("\n\n ENCODER DECODER COMBINED MODEL SUMMARY \n")
		model.summary()

		utils = ModelUtils()

		edModel = EDModel()
		return edModel.getInferenceModelClass(model.name).get_inference_model(utils.get_inference_entity(model))
		
		
	#We are defining a function below which is the implementation of the inference process (which we covered in the above section):
	def decode_sequence(self, input_seq, encoder_model, decoder_model, modelName):
		# Encode the input as state vectors.
		encoderOutputs = encoder_model.predict(input_seq)
		if(Variable.enc_dec_name not in modelName):
			encoderOutputs = self.getEncoderOutputs(encoderOutputs, modelName)

		# Generate empty target sequence of length 1.
		target_seq = np.zeros((1,1))

		# Chose the 'start' word as the first word of the target sequence
		target_seq[0, 0] = 1
		stop_condition = False
		decoded_sentence = ''
		output_index = []
		label_max_len = super().getLabelMaxSentenceLength()
		while not stop_condition:
			
			decoderOutputs = decoder_model.predict([target_seq] + encoderOutputs)
			
			# Sample a token
			#print(decoder_output[0][0, -1, :])
			#print(decoder_output[1:])
			sampled_token_index = np.argmax(decoderOutputs[0][0, -1, :])
			#print(sampled_token_index)
			#sampled_token = reverse_target_word_index[sampled_token_index+1]
			
			output_index.append(sampled_token_index)
			'''if(sampled_token!='end'):
		    		decoded_sentence += ' '+sampled_token
			
				# Exit condition: either hit max length or find stop word.
		    		if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
		        		stop_condition = True'''
			if(len(output_index) == label_max_len):
				stop_condition = True		
			
			# Update the target sequence (of length 1).
			target_seq = np.zeros((1,1))
			target_seq[0, 0] = sampled_token_index

			# Update internal states
			encoderOutputs[1:] = decoderOutputs[1:]

		return output_index
		#return decoded_sentence
		
		
	def getEncoderOutputs(self, encoderOutputs, modelName):
		finalEncoderOutputs = [encoderOutputs[0]]
		encStateCount = len(encoderOutputs[1:])
		decLayerName = self.getDecLayerName(modelName)
		decEncStateCount = self.getDecEncoderStateCount(decLayerName)
		
		if(encStateCount == 4 and decEncStateCount == 2):
			if(Variable.bidirectionalGru_name in modelName):
				finalEncoderOutputs.append(encoderOutputs[1])
				finalEncoderOutputs.append(encoderOutputs[3])
			else:
				finalEncoderOutputs.append(encoderOutputs[1]+encoderOutputs[3])
				finalEncoderOutputs.append(encoderOutputs[2]+encoderOutputs[4])
				
		elif(encStateCount == 4 and decEncStateCount == 1):
			finalEncoderOutputs.append(encoderOutputs[1]+encoderOutputs[3])

		elif(encStateCount == 2 and decEncStateCount == 4):
			if(Variable.bidirectionalGru_name in modelName):
				finalEncoderOutputs.append(encoderOutputs[1])
				finalEncoderOutputs.append(encoderOutputs[1])
				finalEncoderOutputs.append(encoderOutputs[2])
				finalEncoderOutputs.append(encoderOutputs[2])
			else:
				finalEncoderOutputs.append(encoderOutputs[1])
				finalEncoderOutputs.append(encoderOutputs[2])
				finalEncoderOutputs.append(encoderOutputs[1])
				finalEncoderOutputs.append(encoderOutputs[2])
				
		elif(encStateCount == 2 and decEncStateCount == 2):
			if(decLayerName == Variable.lstm_name):
				finalEncoderOutputs.append(encoderOutputs[1]+encoderOutputs[2])
				finalEncoderOutputs.append(encoderOutputs[1]+encoderOutputs[2])
			else:
				finalEncoderOutputs.append(encoderOutputs[1])
				finalEncoderOutputs.append(encoderOutputs[1])
					
		elif(encStateCount == 2 and decEncStateCount == 1):
			if(Variable.bidirectionalGru_name in modelName):
				finalEncoderOutputs.append(encoderOutputs[1]+encoderOutputs[2])
			else:
				finalEncoderOutputs.append(encoderOutputs[1])
				
		elif(encStateCount == 1):
			for i in range(decEncStateCount):
				finalEncoderOutputs.append(encoderOutputs[1])
		
		return finalEncoderOutputs	
			
	
	def getDecEncoderStateCount(self, declayerName):
		#dec_layer_name = dec_layer.__class__.__name__
		if(declayerName == Variable.gru_name):
			return 1
		elif(declayerName == Variable.lstm_name):
			return 2
		elif(declayerName == Variable.bidirectionalGru_name):
			return 2
		else:	
			return 4
			
			
	def startNewPredict(self, sentence):
		print("\nNEW PREDICT START")
		super().setAllConvertData()
		print()
		super().clearAfterTestVariable()		
		super().modifyText([sentence], Variable.TYPE_EXTRAS_TEST)
		print()		
		self.convertAndPredict()
		
		
	def runGradio(self):
		super().runGradio(self,self.startNewPredict)
