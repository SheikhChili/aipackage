# import
import threading
import time
import tensorflow as tf
import numpy as np
# import ktrain
# import gradio as gr
# from ktrain import text
from aipackage.dlpackage.CustomMetrics import CustomMetrics
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.StoreData import InputOutputStream
from aipackage.dlpackage.DataConversion import DataConversion
from aipackage.customThreading import ThreadWithReturnValue
from aipackage.dlpackage.Entities import AccuracyEntity, HyperParameterEntity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Repository:

    # init
    def __init__(self):
        self.type = None
        self.textType = None
        self.userMetrics = None
        self.wantTrainAllData = False
        self.wantTestAllData = False
        self.storedModelFileNameArray = []
        self.datasetLocation = ""

        # initialize
        self.oriPreprocessTrainSentenceArray = []
        self.oriPreprocessTestSentenceArray = []
        self.oriLabelArray = []

    def update_dataset_location(self, dataset_location):
        self.datasetLocation = dataset_location

    def set_type(self, type_value):
        self.type = type_value

    def get_type(self):
        return self.type

    def get_train_all_data(self):
        return self.wantTrainAllData

    def get_test_all_data(self):
        return self.wantTestAllData

    # getter
    def get_ori_preprocess_train_sentence_array(self):
        return self.oriPreprocessTrainSentenceArray

    def get_ori_preprocess_test_sentence_array(self):
        return self.oriPreprocessTestSentenceArray

    def get_ori_label_array(self):
        return self.oriLabelArray

    def set_train_all_data(self, train_all_data):
        self.wantTrainAllData = train_all_data

    def set_test_all_data(self, test_all_data):
        self.wantTestAllData = test_all_data

    def set_all_train_data(self):
        self.set_train_label_data()

    # self.setAllConvertData()

    # SETTER
    def set_ori_preprocess_train_sentence_array(self, array):
        self.oriPreprocessTrainSentenceArray = array

    def set_ori_preprocess_test_sentence_array(self, array):
        self.oriPreprocessTestSentenceArray = array

    def set_ori_label_array(self, array):
        self.oriLabelArray = array

    def read_and_get_ori_train_data(self):
        return self.read_preprocess_data(Variable.fileNamePrefix + Variable.trainPreprocessFileName)

    def set_train_label_data(self):
        self.set_ori_preprocess_train_sentence_array(self.read_and_get_ori_train_data())
        self.set_ori_label_array(self.read_preprocess_data(Variable.fileNamePrefix + Variable.labelFileName))

    def set_all_test_data(self):
        file_name_prefix = Variable.fileNamePrefix
        self.set_ori_preprocess_test_sentence_array(
            self.read_preprocess_data(file_name_prefix + Variable.testPreprocessFileName))

    # self.setAllConvertData()

    def is_data_present(self):
        file_name_prefix = Variable.preprocessFolderName + Variable.locationSeparator
        return self.check_file_exist(file_name_prefix + Variable.trainPreprocessFileName)

    @staticmethod
    def get_train_file_name():
        return Variable.locationSeparator + Variable.featherFolderName + Variable.featherTrainFileName

    @staticmethod
    def get_test_file_name():
        return Variable.locationSeparator + Variable.featherFolderName + Variable.featherTestFileName

    def get_train_data(self, folder_name):
        filename = self.datasetLocation + folder_name + self.get_train_file_name()
        io = InputOutputStream()
        return io.get_stored_model_data(filename)

    def get_test_data(self, folder_name):
        filename = self.datasetLocation + folder_name + self.get_test_file_name()
        io = InputOutputStream()
        return io.get_stored_model_data(filename)

    def convert_train_data_to_feather(self, folder_name):
        actual_folder_path = self.datasetLocation + folder_name + Variable.locationSeparator
        io = InputOutputStream()
        io.check_and_save_as_feather(actual_folder_path, Variable.csvTrainFileName, Variable.featherTrainFileName)

    def convert_test_data_to_feather(self, folder_name):
        actual_folder_path = self.datasetLocation + folder_name + Variable.locationSeparator
        io = InputOutputStream()
        io.check_and_save_as_feather(actual_folder_path, Variable.csvTestFileName, Variable.featherTestFileName)

    def convert_data_to_feather(self, folder_name):
        self.convert_train_data_to_feather(folder_name)
        self.convert_test_data_to_feather(folder_name)

    @staticmethod
    def get_stored_model_data(filename):
        io = InputOutputStream()
        return io.get_stored_model_data(filename)

    def update_user_metrics(self, user_metrics):
        self.userMetrics = user_metrics

    @staticmethod
    def read_preprocess_data(filename):
        return InputOutputStream().get_preprocess_array(filename)

    def get_metrics(self):
        metrics = CustomMetrics()
        metrics_array = metrics.get_nlp_metrics(self.type)
        if self.userMetrics is not None and self.userMetrics not in metrics_array:
            metrics_array.append(self.userMetrics)
        return metrics_array

    @staticmethod
    def normalize_data(data):
        min_max_scaler = StandardScaler()
        return min_max_scaler.fit_transform(data)

    @staticmethod
    def get_xy_data(x, y):
        return np.array(x), np.array(y)

    def store_data(self, file_name_prefix, io_stream):

        t1 = threading.Thread(target=io_stream.store_preprocess_array, args=(
            file_name_prefix + Variable.trainPreprocessFileName, self.get_ori_preprocess_train_sentence_array()))
        t2 = threading.Thread(target=io_stream.store_preprocess_array, args=(
            file_name_prefix + Variable.testPreprocessFileName, self.get_ori_preprocess_test_sentence_array()))

        # start array storing thread
        t1.start()
        t2.start()
        # join
        t1.join()
        t2.join()

    def get_data_limit(self, data_type):
        if data_type == Variable.TYPE_EXTRAS_TEST:
            return self.get_test_data_limit()
        else:
            return self.get_train_data_limit()

    def get_train_data_limit(self):
        train_data_limit = Variable.trainDataLimit
        if self.wantTrainAllData:
            train_data_limit = len(self.get_ori_preprocess_train_sentence_array())
        return train_data_limit

    def get_test_data_limit(self):
        test_data_limit = Variable.testDataLimit
        if self.wantTestAllData:
            test_data_limit = len(self.get_ori_preprocess_test_sentence_array())
        return test_data_limit

    @staticmethod
    def check_file_exist(filename):
        return InputOutputStream().check_file_exist(filename)

    def create_and_write_acc_model(self, acc_entity, file_name=Variable.modelFileName):
        self.write_acc_model(self.create_result_array(acc_entity), file_name)

    @staticmethod
    def write_acc_model(array, file_name):
        io = InputOutputStream()
        io.write_acc_model(array, file_name)

    @staticmethod
    def create_result_array(acc_entity):
        data_conversion = DataConversion()
        return data_conversion.create_result_array(acc_entity)

    @staticmethod
    def write_model_result(array):
        io = InputOutputStream()
        io.write_model_result(array)

    def create_dl_folder(self, folder_path):
        dl_folder_path = Variable.allPredictModelFolderName + Variable.locationSeparator + folder_path
        self.check_and_create_dir(dl_folder_path)

    @staticmethod
    def check_and_create_dir(folder_name):
        io = InputOutputStream()
        io.check_and_create_dir(folder_name)

    '''def createSubmissionArray(self, subEntity):
        dataConversion = DataConversion()
        return dataConversion.createSubmissionArray(subEntity)	'''

    @staticmethod
    def get_max_score_and_val_score():
        io = InputOutputStream()
        df = io.get_stored_model_data(Variable.modelFileName)
        if df.empty:
            score, val_score = 0, 0
        else:
            val_score = max(df[Variable.valScoreName])
            result_df = df[df[Variable.valScoreName] == val_score]
            score = max(result_df[Variable.scoreName])
        return score, val_score

    @staticmethod
    def get_stored_model_file_name_array(file_name=Variable.modelFileName):
        io = InputOutputStream()
        return io.get_stored_model_file_name_array(file_name)

    def update_stored_model_file_name_array(self):
        self.storedModelFileNameArray = self.get_stored_model_file_name_array()

    @staticmethod
    def check_and_create_related_dirs():
        io = InputOutputStream()
        io.check_and_create_all_dirs()

    def run_model(self, hyper_parameter_entity, hyper_parameter_tuning):
        t1 = threading.Thread(target=self.start_model_search, args=([hyper_parameter_entity, hyper_parameter_tuning]))
        t1.start()
        t1.join()

    # models
    def start_model_search(self, hyper_parameter_entity, hyper_parameter_tuning):
        start = time.time()
        print(hyper_parameter_entity.modelClassName + " Start --------- ")

        self.create_dl_folder(hyper_parameter_entity.modelClassName)
        self.run_all_search(hyper_parameter_entity, hyper_parameter_tuning)

        end = time.time()
        print(hyper_parameter_entity.modelClassName + " Finish. In time of seconds = ", int(end - start), "\n\n\n")

    def run_all_search(self, hyper_parameter_entity, hyper_parameter_tuning):
        self.check_and_run_search(hyper_parameter_tuning, hyper_parameter_entity, Variable.typeBayes)
        self.check_and_run_search(hyper_parameter_tuning, hyper_parameter_entity, Variable.typeHyperBand)
        self.check_and_run_search(hyper_parameter_tuning, hyper_parameter_entity, Variable.typeRandom)

    @staticmethod
    def get_model_name(model_class_name, search_name):
        return model_class_name + Variable.filenameSeparator + search_name

    def check_and_run_search(self, hyper_parameter_tuning, hyper_parameter_entity, search_type):
        hyper_parameter_entity.modelName = self.get_model_name(hyper_parameter_entity.modelClassName, search_type)
        if hyper_parameter_entity.modelName in self.storedModelFileNameArray:
            return
        print("\n\n\n\n")
        print(hyper_parameter_entity.modelName + " AND SEARCH TYPE = " + search_type + " Start --------- ")
        search_folder_path = hyper_parameter_entity.modelClassName + Variable.locationSeparator + search_type
        self.create_dl_folder(search_folder_path)

        search_func = hyper_parameter_tuning.start_bayesian_search
        if search_type == Variable.typeHyperBand:
            search_func = hyper_parameter_tuning.start_hyper_band_search
        elif search_type == Variable.typeRandom:
            search_func = hyper_parameter_tuning.start_random_search

        tf.keras.backend.clear_session()
        t11 = ThreadWithReturnValue(target=search_func, args=([hyper_parameter_entity, search_folder_path]))
        t11.start()
        self.create_and_write_acc_model(t11.join())
        print(hyper_parameter_entity.modelName + " AND SEARCH TYPE = " + search_type + " FINISHED --------- ")

    @staticmethod
    def get_best_model():
        io = InputOutputStream()
        return io.get_best_model()

    @staticmethod
    def get_all_saved_model_file_name():
        io = InputOutputStream()
        return io.get_all_saved_model()

    @staticmethod
    def get_loaded_model(model_file_name):
        io = InputOutputStream()
        return io.get_loaded_model(model_file_name)

    @staticmethod
    def is_model_present():
        io = InputOutputStream()
        return io.is_model_present()

    @staticmethod
    def get_full_model_name(model_file_name):
        return model_file_name.split(Variable.locationSeparator)[-1]

    def process_and_set_test_data(self):
        self.set_all_test_data()
        if not self.get_ori_preprocess_test_sentence_array():
            if not self.get_ori_preprocess_train_sentence_array():
                self.set_ori_preprocess_test_sentence_array(self.read_and_get_ori_train_data())
            else:
                self.set_ori_preprocess_test_sentence_array(
                    self.get_ori_preprocess_train_sentence_array()[:self.get_test_data_limit()])

    @staticmethod
    def show_eli5(model, x_test, columns):
        # Return an explanation of estimator parameters (weights).
        eli5.explain_weights(model)
        # Return an explanation of an estimator prediction.
        eli5.explain_prediction(model, x_test)
        # Return an explanation of estimator parameters (weights) as an IPython.display.HTML object. Use this
        # function to show classifier weights in IPython.
        eli5.show_weights(model)
        # Return an explanation of estimator prediction as an IPython.display.HTML object. Use this function to show
        # information about classifier prediction in IPython.
        eli5.show_prediction(model, x_test, feature_names=columns, show_feature_values=True)

    '''@TODO FOR NOW WE WILL COMMENT BECAUSE IT TRAINS ITS OWN MODEL NOT USING OUR MODEL 
    def showEbm(self, X, Y):
        X_train,X_val,Y_train,Y_val = self.getSplittedData(X, Y)
        
        ############## create EBM model #############
        ebm = ExplainableBoostingClassifier()
        ebm.fit(X_train, Y_train)
        
        ############## visualizations #############
        # Generate global explain ability visuals
        global_exp=ebm.explain_global()
        show(global_exp)
        
        # Generate local explain ability visuals
        ebm_local = ebm.explain_local(X, Y)
        show(ebm_local)
        
        # Generate EDA visuals 
        hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
        show(hist)
        
        # Package it all in one Dashboard , see image below
        show([hist, ebm_local, ebm_perf,global_exp], share_tables=True)'''

    def show_dalex(self, model, x, y, x_test):
        explainer = dx.Explainer(model, x, y)  # create explainer from Dalex
        ############## visualizations #############
        # Generate importance plot showing top 30
        explainer.model_parts().plot(max_vars=30)
        # Generate ROC curve for xgboost model object
        print(self.type.lower())
        explainer.model_performance(model_type=self.type.lower()).plot(geom='roc')
        train = x[79]
        # Generate breakdown plot
        explainer.predict_parts(train).plot(max_vars=15)
        # Generate SHAP plot
        explainer.predict_parts(train, type="shap").plot(min_max=[0, 1], max_vars=15)
        # Generate breakdown interactions plot
        explainer.predict_parts(train, type='break_down_interactions').plot(max_vars=20)
        # Generate residual plots
        explainer.model_performance(model_type=self.type.lower()).plot()
        # Generate PDP plots for all variables
        explainer.model_profile(type='partial', label="pdp").plot()
        # Generate Accumulated Local Effects plots for all variables
        explainer.model_profile(type='ale', label="pdp").plot()
        # Generate Individual Conditional Expectation plots for worst texture variable
        # explainer.model_profile(type = 'conditional', label="conditional",variables="worst texture")
        # Generate lime breakdown plot
        explainer.predict_surrogate(train).plot()

        ####### start Arena dashboard #############
        # create empty Arena
        arena = dx.Arena()
        # push created explainer
        arena.push_model(explainer)
        # push whole test dataset (including target column)
        arena.push_observations(pd.DataFrame(x_test))
        # run server on port 9294
        arena.run_server(port=9291)

    def show_explainer_dashboard(self, model, x, y, file_name, columns):
        # Create the explainer object
        x = pd.DataFrame(x, columns=columns)
        y = pd.DataFrame(y)
        if self.type == Variable.typeRegress:
            explainer = RegressionExplainer(model, x, y, model_output='logodds')
        else:
            explainer = ClassifierExplainer(model, x, y, model_output='logodds')

        # Create individual component plants using In explainer
        ie = InlineExplainer(explainer)
        # SHAP overview
        ie.shap.overview()
        # Generate Decision plot
        # SHAP interactions
        ie.shap.interaction_dependence()
        # Model Stats
        ie.classifier.model_stats()
        # SHAP contribution
        ie.shap.contributions_graph()
        # SHAP dependence
        ie.shap.dependence()
        db = ExplainerDashboard(explainer,
                                title=file_name,  # defaults to "Model Explainer"
                                shap_interaction=False,  # you can switch off tabs with bools
                                )
        db.run(port=8805)

    @staticmethod
    def show_shapash(model, x, y, x_test, file_name):
        # create explainer
        xpl = SmartExplainer()
        xpl.compile(x=pd.DataFrame(x_test), model=model)
        # Creating Application
        # app = xpl.run_app(title_story=file_name)

        ############## visualizations #############
        # feature importance based on SHAP
        xpl.plot.features_importance()
        # contributions plot
        # xpl.plot.contribution_plot("worst concave points")
        # Local explanation
        xpl.plot.local_plot(index=79)
        # compare plot
        xpl.plot.compare_plot(index=[x_test.index[79], x_test.index[80]])
        # Interactive interactions widget
        xpl.plot.top_interactions_plot(nb_top_interactions=5)
        # save contributions
        predictor = xpl.to_smartpredictor()
        predictor.add_input(x=x, ypred=y)
        detailed_contributions = predictor.detail_contributions()
        print(detailed_contributions)

    @staticmethod
    def show_lime(model, x, x_test, columns):
        # X_test.columns
        ############## create explainer ###########
        # we use the dataframes splits created above for SHAP
        explainer = lime.lime_tabular.LimeTabularExplainer(x_test, feature_names=columns, verbose=True)
        ############## visualizations #############
        exp = explainer.explain_instance(x[79], model.predict_proba, num_features=len(columns))
        exp.show_in_notebook(show_table=True)

    @staticmethod
    def show_shap(model, x, columns):
        # Generate the Tree explainer and SHAP values
        # model =  model.best_estimator_
        # explainer = shap.TreeExplainer(model)
        explainer = shap.KernelExplainer(model.predict, x)
        shap_values = explainer.shap_values(x)
        expected_value = explainer.expected_value
        ############## visualizations #############
        # Generate summary dot plot
        shap.summary_plot(shap_values, x, title="SHAP summary plot")
        # Generate summary bar plot
        shap.summary_plot(shap_values, x, plot_type="bar")
        # Generate waterfall plot
        shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[79], features=x.loc[79, :],
                                               feature_names=columns, max_display=15, show=True)
        # Generate dependence plot
        # shap.dependence_plot("worst concave points", shap_values, X, interaction_index="mean concave points")
        # Generate multiple dependence plots
        for name in X_train.columns:
            shap.dependence_plot(name, shap_values, x)
        # shap.dependence_plot("worst concave points", shap_values, X, interaction_index="mean concave points")
        # Generate force plot - Multiple rows
        shap.force_plot(explainer.expected_value, shap_values[:100, :], x[:100, :])
        # Generate force plot - Single
        shap.force_plot(explainer.expected_value, shap_values[0, :], x[0, :])
        # Generate Decision plot
        shap.decision_plot(expected_value, shap_values[79], link='logit', features=x[79, :],
                           feature_names=columns, show=True, title="Decision Plot")

    def run_explainable_ai(self, model, x, y, x_test, file_name, columns):
        # columns = list(X.columns)
        print(columns)
        # 1. ELI5
        print("\n ELI5 STARTED ------- \n")
        # self.showEli5(model, X_test, columns)
        print("\n ELI5 FINISHED ------- \n")

        # 2. Explainable Boosting Machines (EBM)
        print("\n Explainable Boosting Machines (EBM) STARTED ------- \n")
        # self.showEbm()	#TODO CREATE NEW MODEL WITHOUT USING TRAINED MODEL
        print("\n Explainable Boosting Machines (EBM) FINISHED ------- \n")

        # 3. Dalex
        print("\n Dalex STARTED ------- \n")
        # self.showDalex(model, X, Y, X_test)
        print("\n Dalex FINISHED ------- \n")

        # 4. ExplainerDashboard
        print("\n ExplainerDashboard STARTED ------- \n")
        # self.showExplainerDashboard(model, X, Y, fileName, columns)
        print("\n ExplainerDashboard FINISHED ------- \n")

        # 5. Shapash
        print("\n SHAPASH STARTED ------- \n")
        self.show_shapash(model, x, y, x_test,
                          file_name)  # TODO THROWING ERROR ValueError: model not supported by shapash, please compute
        # contributions by yourself before using shapash
        print("\n SHAPASH FINISHED ------- \n")

        # 6. Lime
        print("\n LIME STARTED ------- \n")
        # self.showLime(model, X, X_test, columns)
        print("\n LIME FINISHED ------- \n")

        # 7. SHAP
        print("\n SHAP STARTED ------- \n")
        # self.showSHAP(model, X, columns )
        print("\n SHAP FINISHED ------- \n")

    def start_predict(self, sub_entity=None, predict_label_dict=None):
        x_test = self.testData

        # model = self.getBestModel()
        # model.summary()

        # pred = predict(x_test, model)

        # self.run_explainable_ai(model, self.X, self.Y, x_test, self.folderName, self.getTrainColumns(train,
        # labelName))

        pred = self.predict_all_model(x_test)

        if sub_entity is not None:
            if predict_label_dict is not None:
                pred = [predict_label_dict[value] for value in pred]
            sub_entity.predictions = pred
            self.submit_predictions(sub_entity)
        else:
            return pred[0]

    def predict_all_model(self, x_test):
        stored_predict_name_array = self.get_stored_model_file_name_array(Variable.predictFileName)

        for modelFileName in self.get_all_saved_model_file_name():
            full_model_name = self.get_full_model_name(modelFileName)

            if full_model_name in stored_predict_name_array:
                continue

            print("\n\n" + full_model_name + " _ _    PREDICT THE DATA")

            model = self.get_loaded_model(modelFileName)
            model.summary()

            self.predict(x_test, model)
            self.save_predict_file_name(full_model_name)
        return []

    @staticmethod
    def predict(x_test, model):
        print("\n\nPREDICT THE DATA")
        print("\nTEST SET SIZE = ", len(x_test), "\n")
        predictor = ktrain.get_predictor(model)
        predictor.explain(x_test)
        return model.predict(x_test)

    def save_predict_file_name(self, full_model_name):
        self.create_and_write_acc_model(AccuracyEntity(full_model_name), Variable.predictFileName)

    @staticmethod
    def create_submission_array(sub_entity):
        data_conversion = DataConversion()
        return data_conversion.create_submission_array(sub_entity)

    def submit_predictions(self, sub_entity):
        io = InputOutputStream()
        array = self.create_submission_array(sub_entity)
        io.write_csv(self.get_submission_file_name(sub_entity.fileName), array, sub_entity.fields)

    @staticmethod
    def get_submission_file_name(file_name):
        if file_name is None:
            return Variable.submissionFileName
        else:
            return file_name

    def evaluate_and_update_nlp_model(self):
        x, y = self.get_xy_data()
        x_train, x_val, y_train, y_val = self.get_splitted_data(x, y)
        for modelFileName in self.get_all_saved_model_file_name():
            t11 = ThreadWithReturnValue(target=self.evaluate_model_and_update_score,
                                        args=([modelFileName, x_train, x_val, y_train, y_val]))
            t11.start()
            t11.join()

    def evaluate_model_and_update_score(self, model_file_name, x_train, x_val, y_train, y_val):
        start = time.time()
        print(model_file_name + " Start Evaluation --------- ")

        model = self.get_loaded_model(model_file_name)
        model.compile(optimizer="adam")
        t11 = ThreadWithReturnValue(target=self.evaluate_model, args=(model, x_train, y_train))
        t12 = ThreadWithReturnValue(target=self.evaluate_model, args=(model, x_val, y_val))
        t11.start()
        t12.start()
        score = t11.join()
        val_score = t12.join()
        print(score)
        print(val_score)

        end = time.time()
        print(model_file_name + " Finish Evaluation. In time of seconds = ", int(end - start), "\n\n\n")

    @staticmethod
    def get_splitted_data(x, y):
        # TRAIN TEST SPLIT DATASET
        return train_test_split(x, y, train_size=0.90, random_state=1)

    @staticmethod
    def evaluate_model(model, x, y):
        return model.evaluate(x, y)
