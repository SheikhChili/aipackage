class InferenceEntity():
	
	def __init__(self, infEncoderEntity, infDecoderEntity, attention_layer, dense_layer, input_length, units):
		self.infEncoderEntity = infEncoderEntity
		self.infDecoderEntity = infDecoderEntity
		self.attention_layer = attention_layer
		self.dense_layer = dense_layer
		self.input_length = input_length
		self.units = units
		
		
class EmbeddingEntity():

	def __init__(self, input_length, vocab_size = None, embedding_dim = None, embedding_matrix = None):
		self.vocab_size = vocab_size
		self.embedding_matrix = embedding_matrix
		self.embedding_dim = embedding_dim 
		self.input_length = input_length
		
		
class InfEncoderEntity():
	
	def __init__(self, encoder_input, encoder_rnn_output):
		self.encoder_input = encoder_input
		self.encoder_rnn_output = encoder_rnn_output	
		
class InfDecoderEntity():
	
	def __init__(self, decoder_input, decoder_embedding_layer, decoder_rnn_layer):
		self.decoder_input = decoder_input
		self.decoder_embedding_layer = decoder_embedding_layer
		self.decoder_rnn_layer = decoder_rnn_layer
