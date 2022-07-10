# IMPORTS
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.StoreData import InputOutputStream
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband, Objective
# from tensorgram import TensorGram
from aipackage.dlpackage.CustomMetrics import CustomEarlyStopping, CustomLearningRateScheduler
from aipackage.dlpackage.Entities import AccuracyEntity


class HyperParameterTuning:

    def __init__(self, dataset_entity):
        self.best_score = 0
        self.best_val_score = 0
        self.datasetEntity = dataset_entity
        self.type = ""

    def set_type(self, type_value):
        self.type = type_value

    def set_scores(self, score, val_score):
        self.best_score = score
        self.best_val_score = val_score

    def start_random_search(self, hyper_parameter_entity, folder_path):
        directory_name = Variable.ranSearchFolderName + Variable.locationSeparator + 'rs' + str(
            hyper_parameter_entity.num)

        tuner_rs = RandomSearch(
            hyper_parameter_entity.model,
            objective=self.get_search_objective(),
            max_trials=Variable.max_trials,
            executions_per_trial=Variable.executions_per_trial,
            directory=directory_name,
            project_name=hyper_parameter_entity.modelClassName
        )

        return self.start_search(tuner_rs, hyper_parameter_entity.modelName, folder_path)

    def start_bayesian_search(self, hyper_parameter_entity, folder_path):
        directory_name = Variable.baySearchFolderName + Variable.locationSeparator + 'bo' + str(
            hyper_parameter_entity.num)

        tuner_bo = BayesianOptimization(
            hyper_parameter_entity.model,
            objective=self.get_search_objective(),
            max_trials=Variable.max_trials,
            executions_per_trial=Variable.executions_per_trial,
            directory=directory_name,
            project_name=hyper_parameter_entity.modelClassName
        )

        return self.start_search(tuner_bo, hyper_parameter_entity.modelName, folder_path)

    def start_hyper_band_search(self, hyperParameterEntity, folderPath):
        directory_name = Variable.hpSearchFolderName + Variable.locationSeparator + 'bo' + str(hyperParameterEntity.num)

        tuner_hp = Hyperband(
            hyperParameterEntity.model,
            objective=self.get_search_objective(),
            max_epochs=Variable.max_trials,
            hyperband_iterations=Variable.executions_per_trial,
            directory=directory_name,
            project_name=hyperParameterEntity.modelClassName
        )

        return self.start_search(tuner_hp, hyperParameterEntity.modelName, folderPath)

    @staticmethod
    def get_search_objective():
        # return Objective(Variable.objective, direction="min")
        return Variable.objective

    @staticmethod
    def get_distribute_strategy():
        return Variable.distribution_strategy

    def start_search(self, hpSearch, model_name, folderPath):
        x = self.datasetEntity.X
        y = self.datasetEntity.Y

        batch_size = Variable.batch_size
        callbacks = self.get_callback(model_name)

        """ if self.type == Variable.typeClassification:
			X_train,X_val,Y_train,Y_val = self.getSplittedData(x, y)
		
			# Wrap data in Dataset objects.
			train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
			val_data = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
			
			# The batch size must now be set on the Dataset objects.
			train_data = train_data.batch(batch_size)
			val_data = val_data.batch(batch_size)
			
			# Disable AutoShard.
			options = tf.data.Options()
			options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
			train_data = train_data.with_options(options)
			val_data = val_data.with_options(options)
		
		
			hpSearch.search(train_data, validation_data = val_data, epochs=Variable.epochs, callbacks = callbacks, 
			batch_size=Variable.batch_size)
		else:"""

        hpSearch.search(x, y, validation_split=0.2, epochs=Variable.epochs, callbacks=callbacks,
                        batch_size=Variable.batch_size)
        print("\n\n-------------------------------------------------------------------------------------------")

        best_model = hpSearch.get_best_models(num_models=1)[0]
        print("\n\n", hpSearch.results_summary())
        self.save_all_model(model_name, folderPath, best_model)

        # return result
        del best_model
        return AccuracyEntity(model_name)

    @staticmethod
    def get_callback(model_name):
        log_dir = "./logs/" + model_name
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        return [CustomEarlyStopping(), CustomLearningRateScheduler().get_learning_rate_scheduler(),
                tensorboard_callback]  # , TensorGram(model_name, "1329984812")]

    def can_save_model(self, score, val_score):
        if self.best_val_score < val_score:
            return True
        if self.best_val_score == val_score:
            if self.best_score < score or self.best_score == score:
                return True

    def clear_variable(self):
        del self.datasetEntity

    def check_and_save_model(self, score, val_score, model_name, folderPath, model):
        self.save_all_model(model_name, folderPath, model)
        if self.can_save_model(score, val_score):
            self.save_best_model(model_name, model)
            self.best_score = score
            self.best_val_score = val_score

    @staticmethod
    def save_best_model(model_name, model):
        io = InputOutputStream()
        io.save_best_model(model_name, model)

    @staticmethod
    def save_all_model(model_name, folderPath, model):
        io = InputOutputStream()
        io.save_all_model(model_name, folderPath, model)

    @staticmethod
    def get_splited_data(X, Y):
        # TRAIN TEST SPLIT DATASET
        return train_test_split(X, Y, train_size=0.70, random_state=1)

    @staticmethod
    def create_dir(folder_name):
        io = InputOutputStream()
        io.check_and_create_dir(folder_name)
