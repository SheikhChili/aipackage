class ModelEntity():
	"""A class for holding an article content"""
  
	# Attributes Declaration 
    	# using Type Hints 
	def __init__(self, target_size, final_activation, loss, optimizer, metrics, modelName = ""):
	    	self.target_size = target_size
	    	self.final_activation = final_activation
	    	self.loss = loss
	    	self.optimizer = optimizer
	    	self.metrics = metrics
	    	self.modelName = modelName
	    	
	    	
class DatasetEntity():
	
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

		
class AccuracyEntity():
	
	def __init__(self, filename, score = 0, val_score = 0):
		self.filename = filename
		self.score = score
		self.val_score = val_score
		
		
class HyperParameterEntity():
	
	def __init__(self, model, num, modelClassName = "", modelName = ""):
		self.model = model
		self.num = num
		self.modelName = modelName
		self.modelClassName = modelClassName
		
		
class SubmissionEntity():
	def __init__(self, predictions = [], id_ = [], id_2 = [], id_3 = [], fields=[], fileName = None):
		self.predictions = predictions
		self.id_ = id_
		self.id_2 = id_2
		self.id_3 = id_3
		self.fields = fields
		self.fileName = fileName		
