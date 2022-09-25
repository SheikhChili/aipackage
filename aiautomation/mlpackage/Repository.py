# IMPORT
import threading
import time

# import dalex as dx
# EDA
import dtale
import eli5
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# EXPLAINABLE
import shap
import sweetviz as sv
from aiautomation.customThreading import ThreadWithReturnValue
from aiautomation.mlpackage.CustomMetrics import CustomMetrics
from aiautomation.mlpackage.DataConversion import DataConversion
from aiautomation.mlpackage.PackageVariable import Variable
from aiautomation.mlpackage.StoreData import InputOutputStream
# IBM EX_AI
from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier
from aix360.algorithms.ted.TED_Cartesian import TED_CartesianExplainer
# from shapash.explainer.smart_explainer import SmartExplainer
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, RegressionExplainer
from explainerdashboard import InlineExplainer
from pandas_profiling import ProfileReport
from dataprep.eda import create_report
from autoviz.AutoViz_Class import AutoViz_Class
from pandas_visual_analysis import VisualAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


class Repository:

    def __init__(self):
        self.type = Variable.typeClassification
        self.storedModelFileNameArray = []
        self.storedEdaFileNameArray = []
        self.scoringDict = {}
        self.userScoringDict = None
        self.datasetLocation = Variable.datasetLocation
        self.update_dataset_location()
        self.io = InputOutputStream()
        self.X = []
        self.Y = []
        self.folderName = ""
        self.actualFolderName = ""

    def get_xy_data(self):
        return self.X, self.Y

    def update_xy_data(self, x, y):
        x[x == -np.inf] = 0
        x[x == np.inf] = 0
        self.X = x
        self.Y = y

    def set_type(self, type_value):
        self.type = type_value

    @staticmethod
    def convert_data(dataset):
        data_conversion = DataConversion()
        return data_conversion.convert_to_num(dataset)

    @staticmethod
    def normalize_data(dataset):
        data_conversion = DataConversion()
        return data_conversion.normalize_data(dataset)

    @staticmethod
    def synthesize_data(train, df, label_name):
        data_conversion = DataConversion()
        return data_conversion.synthesize_data(train, df, label_name)

    def create_model_folder(self, folder_path):
        self.io.create_model_folder(folder_path)

    def run_all_search_and_all_scoring(self, hyper_parameter_tuning, model_entity, search_type):
        search_folder_path = model_entity.model_name + Variable.locationSeparator + search_type
        self.create_model_folder(search_folder_path)
        search_func = hyper_parameter_tuning.model_grid_fit
        if search_type == Variable.typeBayes:
            search_func = hyper_parameter_tuning.model_bayes_fit
        elif search_type == Variable.typeGene:
            search_func = hyper_parameter_tuning.model_gene_fit
        elif search_type == Variable.typeOptuna:
            search_func = hyper_parameter_tuning.model_optuna_fit
        elif search_type == Variable.typeGA:
            search_func = hyper_parameter_tuning.model_sklearn_genetic_fit
        folder_path = model_entity.model_name + Variable.locationSeparator + search_type
        for scoringKey in self.scoringDict.keys():
            filename = model_entity.model_name + search_type + Variable.fileSeparator + scoringKey
            if filename in self.storedModelFileNameArray:
                continue
            print("Search type = ", search_type, " and metric name = ", scoringKey, flush=True)
            hyper_parameter_tuning.set_scoring(self.scoringDict[scoringKey])

            t11 = ThreadWithReturnValue(target=search_func, args=([model_entity, filename, folder_path]))
            t11.start()
            acc_entity = t11.join()
            acc_entity.metrics = scoringKey
            self.create_and_write_acc_model(acc_entity)

    def create_and_write_acc_model(self, acc_entity):
        self.write_acc_model(self.create_result_array(acc_entity))

    def write_acc_model(self, array):
        self.io.write_acc_model(array)

    def write_eda_feather(self, file_name):
        self.io.write_eda_feather(file_name)

    @staticmethod
    def create_result_array(acc_entity):
        data_conversion = DataConversion()
        return data_conversion.create_result_array(acc_entity)

    @staticmethod
    def create_submission_array(sub_entity):
        data_conversion = DataConversion()
        return data_conversion.create_submission_array(sub_entity)

    def concat_to_dataset_location(self, type_name):
        self.datasetLocation = self.datasetLocation + type_name + Variable.locationSeparator

    def run_all_search(self, model_entity, hyper_parameter_tuning, is_single_label):
        t11 = threading.Thread(target=self.run_all_search_and_all_scoring,
                               args=([hyper_parameter_tuning, model_entity, Variable.typeGrid]))
        t12 = threading.Thread(target=self.run_all_search_and_all_scoring,
                               args=([hyper_parameter_tuning, model_entity, Variable.typeBayes]))
        t13 = threading.Thread(target=self.run_all_search_and_all_scoring,
                               args=([hyper_parameter_tuning, model_entity, Variable.typeGene]))
        t14 = threading.Thread(target=self.run_all_search_and_all_scoring,
                               args=([hyper_parameter_tuning, model_entity, Variable.typeOptuna]))
        t15 = threading.Thread(target=self.run_all_search_and_all_scoring,
                               args=([hyper_parameter_tuning, model_entity, Variable.typeGA]))
        t11.start()
        t12.start()
        t14.start()
        t15.start()
        if is_single_label:
            t13.start()
            t13.join()
            pass
        t11.join()
        t12.join()
        t14.join()
        t15.join()

    '''def getSplitedData(self):
        self.getSplitedData(self.getXYData())'''

    def get_splitted_data(self):
        # TRAIN TEST SPLIT DATASET
        xy_data = self.get_xy_data()
        return train_test_split(xy_data[0], xy_data[1], train_size=0.80, random_state=1)

    @staticmethod
    def get_train_file_name():
        return Variable.locationSeparator + Variable.featherFolderName + Variable.featherTrainFileName

    @staticmethod
    def get_test_file_name():
        return Variable.locationSeparator + Variable.featherFolderName + Variable.featherTestFileName

    def update_dataset_location(self):
        if Variable.isRunAllFileEnabled:
            self.datasetLocation = Variable.tempDatasetLocation
        else:
            self.datasetLocation = Variable.datasetLocation

    def get_train_data(self, folder_name):
        self.folderName = folder_name
        file_name = self.datasetLocation + folder_name + self.get_train_file_name()
        train_df = self.io.get_stored_model_data(file_name)
        print("\n", train_df.shape[0])
        if (Variable.isRunAllFileEnabled and train_df.shape[0] not in range(Variable.runAllFileLimitStart,
                                                                            Variable.runAllFileLimitEnd + 1)):
            raise Exception("DATASET LIMIT EXCEEDED")
        return train_df

    def get_test_data(self, folder_name):
        file_name = self.datasetLocation + folder_name + self.get_test_file_name()
        return self.io.get_stored_model_data(file_name)

    def convert_train_data_to_feather(self, folder_name):
        actual_folder_path = self.datasetLocation + folder_name + Variable.locationSeparator
        self.io.check_and_save_as_feather(actual_folder_path, Variable.csvTrainFileName, Variable.featherTrainFileName)

    def convert_test_data_to_feather(self, folder_name):
        actual_folder_path = self.datasetLocation + folder_name + Variable.locationSeparator
        self.io.check_and_save_as_feather(actual_folder_path, Variable.csvTestFileName, Variable.featherTestFileName)

    def convert_data_to_feather(self, folder_name):
        self.convert_train_data_to_feather(folder_name)
        self.convert_test_data_to_feather(folder_name)
        self.actualFolderName = folder_name

    def get_evaluation_data(self):
        return self.io.get_stored_model_data(Variable.modelFileName)

    @staticmethod
    def get_train_columns(train, label_name):
        train = train.drop([label_name], axis=1)
        return list(train.columns)

    def process_and_get_train_data(self, train, label_name, df=None):
        if df is None:
            df = train[label_name]
            train = train.drop([label_name], axis=1)

        return self.remove_outlier_and_convert_data(train, df)

    def remove_outlier_and_convert_data(self, train, df):
        print("\n\nBEFORE OUTLIER = ", len(train))
        train = self.convert_data(train)
        mask = self.get_outlier_mask(train)
        train = self.normalize_data(train.iloc[mask, :])
        df = df[mask]
        print("AFTER OUTLIER = ", len(train), "\n\n")
        return train, df

    @staticmethod
    def get_outlier_mask(train):
        iso = IsolationForest(contamination=0.1)
        yhat = iso.fit_predict(train)
        return yhat != -1

    @staticmethod
    def check_and_get_test_data(test, train, label_name):
        if not test:
            return train.drop([label_name], axis=1)
        else:
            return test

    def get_max_score_and_val_score(self):
        df = self.io.get_stored_model_data(Variable.modelFileName)
        if df.empty:
            score, val_score = self.get_initial_best_score_value()
        else:
            val_score = self.check_and_get_best_score(df[Variable.valScoreName])
            result_df = df[df[Variable.valScoreName] == val_score]
            score = self.check_and_get_best_score(result_df[Variable.scoreName])
        return score, val_score

    def get_initial_best_score_value(self):
        if self.type == Variable.typeRegress:
            return 100, 100
        else:
            return 0, 0

    def check_and_get_best_score(self, df):
        df = df[df >= 0]
        if self.type == Variable.typeRegress:
            return min(df)
        else:
            return max(df)

    # models
    def run_model(self, model_entity, hyper_parameter_tuning, is_single_label):
        start = time.time()
        print(model_entity.model_name + "Start --------- ")

        self.create_model_folder(model_entity.model_name)
        self.run_all_search(model_entity, hyper_parameter_tuning, is_single_label)

        end = time.time()
        print(model_entity.model_name + " Finish. In time of seconds = ", int(end - start), "\n\n\n")

    def run_all_models(self, hyper_parameter_tuning, model_array, is_single_label=True):
        self.check_and_create_related_dirs()
        self.update_stored_model_file_name_array()
        self.update_scoring_dict()
        hyper_parameter_tuning.update_io(self.io)
        hyper_parameter_tuning.set_actual_folder_name(self.actualFolderName)

        scores = self.get_max_score_and_val_score()
        print("SCORE = ", scores, "\n")
        hyper_parameter_tuning.set_scores(scores[0], scores[1])
        for modelEntity in model_array:
            t1 = threading.Thread(target=self.run_model, args=([modelEntity, hyper_parameter_tuning, is_single_label]))
            t1.start()
            t1.join()

    def show_auto_viz(self, df, file_name, label_name):
        if file_name + Variable.autoviz in self.storedEdaFileNameArray:
            return
        # WILL WORK IN JUPYTER HERE NO VISUALIZATION
        av = AutoViz_Class()
        sep = ","
        autoviz_dir = self.io.storeDataDirName + Variable.edaLocation + Variable.locationSeparator + file_name + \
                      Variable.autoviz + Variable.htmlExtension
        dft = av.AutoViz('', sep, label_name, df, chart_format="html", save_plot_dir=autoviz_dir)
        print("\n")
        '''    sep, which is the separator by which data is separated, by default it is ‘,’.
                target, which is the target variable in the dataset.
                chart_format is the format of the chart displayed.
                max_row_analyzed is used to define the number of rows to be analyzed
                max_cols_analyzed is used to define the number of columns to be analyzed.
                argument depVar which is the dependent variable so that AutoViz creates visualization accordingly.'''
        self.write_eda_feather(file_name + Variable.autoviz)

    def show_pandas_profiling(self, df, file_name):
        if file_name + Variable.pandasProfiling in self.storedEdaFileNameArray:
            return
        profile = ProfileReport(df, title=file_name, explorative=True)
        profile.to_widgets()
        html_filename = self.io.storeDataDirName + Variable.edaLocation + Variable.locationSeparator + file_name + \
                        Variable.pandasProfiling + Variable.htmlExtension
        profile.to_file(html_filename)
        self.write_eda_feather(file_name + Variable.pandasProfiling)

    def show_sweetviz(self, df, file_name):
        if file_name + Variable.sweetviz in self.storedEdaFileNameArray:
            return
        report = sv.analyze(df)
        html_filename = self.io.storeDataDirName + Variable.edaLocation + Variable.locationSeparator + file_name + \
                        Variable.sweetviz + Variable.htmlExtension
        report.show_html(html_filename, open_browser=False)
        self.write_eda_feather(file_name + Variable.sweetviz)

    # As of now , we will comment this because this is not useful
    def show_klib(self, df):
        """'# klib.describe #- functions for visualizing datasets
        klib.cat_plot(df)  # returns a visualization of the number and frequency of categorical features
        klib.corr_mat(df)  # returns a color-encoded correlation matrix
        klib.corr_plot(df)  # returns a color-encoded heatmap, ideal for correlations
        klib.dist_plot(df)  # returns a distribution plot for every numeric feature
        klib.missingval_plot(df)  # returns a figure containing information about missing values

        # klib.clean - functions for cleaning datasets df_clean = klib.data_cleaning(df) df_clean.info() #This reduces the memory data type
        klib.data_cleaning(df)  # performs datacleaning (drop duplicates & empty rows/cols, adjust dtypes,...)
        klib.clean_column_names(df)  # cleans and standardizes column names, also called inside data_cleaning()
        klib.convert_datatypes(df)  # converts existing to more efficient dtypes, also called inside data_cleaning()
        klib.drop_missing(df)  # drops missing values, also called in data_cleaning()
        klib.mv_col_handling(df)  # drops features with high ratio of missing vals based on informational content
        klib.pool_duplicate_subsets(df)  # pools subset of cols based on duplicates with min. loss of information

        # klib. preprocess - functions for data preprocessing (feature selection, scaling, ...)
        # klib.train_dev_test_split(df) # splits a dataset and a label into train, optionally dev and test sets
        klib.feature_selection_pipe()  # provides common operations for feature selection
        klib.num_pipe()  # provides common operations for preprocessing of numerical data
        klib.cat_pipe()  # provides common operations for preprocessing of categorical data
        klib.preprocess.ColumnSelector()  # selects num or cat columns, ideal for a Feature Union or Pipeline
        klib.preprocess.PipeInfo()  # prints out the shape of the data at the specified step of a Pipeline"""
        pass

    @staticmethod
    def show_dtale(df, file_name):
        if file_name + Variable.dtale in self.storedEdaFileNameArray:
            return
        # dtale_file_path = Variable.edaLocation + Variable.locationSeparator + file_name + Variable.dtale +
        # Variable.htmlExtension dtale.offline_chart(df, filepath=dtale_file_path, title=fileName)
        dtale.show(df).open_browser()
        self.write_eda_feather(file_name + Variable.dtale)

    def show_data_prep(self, df, file_name):
        if file_name + Variable.dataPrep in self.storedEdaFileNameArray:
            return
        report = create_report(df, title=file_name)
        report.save(filename=file_name + Variable.dataPrep, to=self.io.storeDataDirName + Variable.edaLocation)
        self.write_eda_feather(file_name + Variable.dataPrep)

    def run_eda(self, file_name, label_name, df, should_use_data_prep=True):
        self.update_stored_eda_file_name_array()
        if df.empty:
            return

        # 1. DTALE
        print("\n DTALE STARTED ------- \n")
        # self.show_dtale(df, file_name)  # OPEN BROWSER
        print("\n DTALE FINISHED ------- \n")

        # 2. PANDAS PROFILING
        print("\n PANDAS PROFILING STARTED ------- \n")
        self.show_pandas_profiling(df, file_name)  # SAVE FILE
        print("\n PANDAS PROFILING FINISHED ------- \n")

        # 3. SWEETVIZ
        print("\n SWEETVIZ STARTED ------- \n")
        self.show_sweetviz(df, file_name)  # SAVE FILE
        print("\n SWEETVIZ FINISHED ------- \n")

        # 4. LUX
        # @TODO: NOT WORKING WILL IMPLEMENT IN FUTURE
        print("\n LUX STARTED ------- \n")
        # print(df)
        print("\n LUX FINISHED ------- \n")

        # 5. DATA PREP
        print("\n DATA PREP STARTED ------- \n")
        if should_use_data_prep:
            self.show_data_prep(df, file_name)  # PACKAGE ERROR
        print("\n DATA PREP FINISHED ------- \n")

        # 6. PANDAS VISUAL ANALYSIS
        print("\n PANDAS VISUAL ANALYSIS STARTED ------- \n")
        # VisualAnalysis(df)
        print("\n PANDAS VISUAL ANALYSIS FINISHED ------- \n")

        # 7. AUTOVIZ
        print("\n AUTOVIZ STARTED ------- \n")
        self.show_auto_viz(df, file_name, label_name)  # PACKAGE ERROR
        print("\n AUTOVIZ FINISHED ------- \n")

        ''''# 8. K L I B E
        print("\n KLIBE STARTED ------- \n")
        self.show_klib(df)  # As of now , we will comment this because this is not useful
        print("\n KLIBE FINISHED ------- \n")'''
        # assert(0==1)

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
        explainer = dx.Explainer(model, x, y)

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
        # create explainer
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

    def show_cem_explainer(self, model, x_test):
        # wrap mnist_model into a framework independent class structure
        mymodel = KerasClassifier(model)

        # initialize explainer object
        explainer = CEMExplainer(mymodel)

        # check model prediction
        print("Predicted class:", mymodel.predict_classes(np.expand_dims(x_test, axis=0)))
        print("Predicted log:", mymodel.predict(np.expand_dims(x_test, axis=0)))

        '''a) Pertinent Negatives (PNs): It identifies a minimal set of features which if altered would change the 
        classification of the original input. For example, in this case if a person’s credit score is increased their 
        loan application status may change from reject to accept. 

 b) Pertinent Positives (PPs) : It  identifies a minimal set of features and their values that are sufficient to 
 yield the original input’s classification. For example, an individual’s loan may still be accepted if the salary was 
 50K as opposed to 100K. '''

        self.per_neg_explanation(explainer, model, x_test)

    def per_neg_explanation(self, explainer, model, x_test, x_train, columns, class_names):
        idx = 1272
        x = x_test[idx].reshape((1,) + x_test[idx].shape)
        print("Computing PN for Sample:", idx)
        print("Prediction made by the model:", model.predict_proba(x))
        print("Prediction probabilities:", class_names[np.argmax(model.predict_proba(x))])
        print("")

        # Obtain Pertinent Negative (PN) explanation
        arg_mode = "PN"  # Find pertinent negative

        arg_max_iter = 1000  # Maximum number of iterations to search for the optimal PN for given parameter settings
        arg_init_const = 10.0  # Initial coefficient value for main loss term that encourages class change
        arg_b = 9  # No. of updates to the coefficient of the main loss term

        arg_kappa = 0.9  # Minimum confidence gap between the PNs (changed) class probability and original class'
        # probability
        arg_beta = 1.0  # Controls sparsity of the solution (L1 loss)
        arg_gamma = 100  # Controls how much to adhere to a (optionally trained) autoencoder
        arg_alpha = 0.01  # Penalizes L2 norm of the solution
        arg_threshold = 0.05  # Automatically turn off features <= arg_threshold if arg_threshold < 1
        arg_offset = 0.5  # the model assumes classifier trained on data normalized
        # in [-arg_offset, arg_offset] range, where arg_offset is 0 or 0.5

        (adv_pn, delta_pn, info_pn) = explainer.explain_instance(np.expand_dims(x, axis=0), arg_mode,
                                                                 model, arg_kappa, arg_b,
                                                                 arg_max_iter, arg_init_const, arg_beta, arg_gamma,
                                                                 arg_alpha, arg_threshold, arg_offset)
        print(info_pn)
        print("\n")

        xpn = adv_pn
        classes = [class_names[np.argmax(model.predict_proba(x))], class_names[np.argmax(model.predict_proba(xpn))],
                   'NIL']

        print("Sample:", idx)
        print("prediction(x)", model.predict_proba(x), class_names[np.argmax(model.predict_proba(x))])
        print("prediction(xpn)", model.predict_proba(xpn), class_names[np.argmax(model.predict_proba(xpn))])

        # X_re = self.rescale(x)  # Convert values back to original scale from normalized
        # Xpn_re = self.rescale(xpn)
        # Xpn_re = np.around(Xpn_re.astype(np.double), 2)

        delta_re = xpn - x
        delta_re = np.around(delta_re.astype(np.double), 2)
        delta_re[np.absolute(delta_re) < 1e-4] = 0

        x3 = np.vstack((x, xpn, delta_re))

        df_re = pd.DataFrame.from_records(x3)  # Create dataframe to display original point, PN and difference (delta)
        df_re[23] = classes

        df_re.columns = columns
        df_re.rename(index={0: 'x', 1: 'X_PN', 2: '(X_PN - x)'}, inplace=True)
        df_ret = df_re.transpose()

        df_ret.style.apply(self.highlight_ce, col='(X_PN - x)', ncols=3, axis=1)

        plt.rcdefaults()
        fi = abs((x - xpn).astype('double')) / np.std(x_train.astype('double'),
                                                      axis=0)  # Compute PN feature importance
        objects = columns[-2::-1]
        y_pos = np.arange(len(objects))
        performance = fi[0, -1::-1]

        plt.barh(y_pos, performance, align='center', alpha=0.5)  # bar chart
        plt.yticks(y_pos, objects)  # Display features on y-axis
        plt.xlabel('weight')  # x-label
        plt.title('PN (feature importance)')  # Heading

        plt.show()  # Display PN feature importance

    @staticmethod
    def highlight_ce(s, col, cols):
        if type(s[col]) != str:
            if s[col] > 0:
                return ['background-color: yellow'] * cols
        return ['background-color: white'] * cols

    def per_pos_explanation(self, explainer, model, x_test, x_train, columns, class_names):
        # Obtain Pertinent Positive (PP) explanation
        # Some interesting user samples to try: 9 11 24
        idx = 9

        x = x_test[idx].reshape((1,) + x_test[idx].shape)
        print("Computing PP for Sample:", idx)
        print("Prediction made by the model:", class_names[np.argmax(model.predict_proba(x))])
        print("Prediction probabilities:", model.predict_proba(x))
        print("")

        arg_mode = 'PP'  # Find pertinent positives
        arg_max_iter = 1000  # Maximum number of iterations to search for the optimal PN for given parameter settings
        arg_init_const = 10.0  # Initial coefficient value for main loss term that encourages class change
        arg_b = 9  # No. of updates to the coefficient of the main loss term
        arg_kappa = 0.2  # Minimum confidence gap between the PNs (changed) class probability and original class'
        # probability
        arg_beta = 10.0  # Controls sparsity of the solution (L1 loss)
        arg_gamma = 100  # Controls how much to adhere to a (optionally trained) auto-encoder
        my_ae_model = None  # Pointer to an auto-encoder
        arg_alpha = 0.1  # Penalizes L2 norm of the solution
        arg_threshold = 0.0  # Automatically turn off features <= arg_threshold if arg_threshold < 1
        arg_offset = 0.5  # the model assumes classifier trained on data normalized
        # in [-arg_offset, arg_offset] range, where arg_offset is 0 or 0.5
        (adv_pp, delta_pp, info_pp) = explainer.explain_instance(x, arg_mode, model, arg_kappa, arg_b,
                                                                 arg_max_iter, arg_init_const, arg_beta, arg_gamma,
                                                                 arg_alpha, arg_threshold, arg_offset)

        print(info_pp)
        print("\n")

        xpp = delta_pp
        classes = [class_names[np.argmax(model.predict_proba(x))], class_names[np.argmax(model.predict_proba(xpp))]]

        print("PP for Sample:", idx)
        print("Prediction(xpp) :", class_names[np.argmax(model.predict_proba(xpp))])
        print("Prediction probabilities for xpp:", model.predict_proba(xpp))
        print("")

        # X_re = rescale(x)  # Convert values back to original scale from normalized
        # adv_pp_re = rescale(adv_pp)
        # Xpp_re = X_re - adv_pp_re
        # Xpp_re = rescale(xpp)
        # Xpp_re = np.around(Xpp_re.astype(np.double), 2)
        xpp[xpp < 1e-4] = 0

        x2 = np.vstack((x, xpp))

        df_pp = pd.DataFrame.from_records(x2.astype('double'))  # Showcase a dataframe for the original point and PP
        df_pp[23] = classes
        df_pp.columns = columns
        df_pp.rename(index={0: 'x', 1: 'X_PP'}, inplace=True)
        df_ppt = df_pp.transpose()

        df_ppt.style.apply(self.highlight_ce, col='X_PP', ncols=2, axis=1)

        # Plot Pertinent Negative (PN) and Pertinent Positive (PP) explanations

        plt.rcdefaults()
        fi = abs(xpp.astype('double')) / np.std(x_train.astype('double'), axis=0)  # Compute PP feature importance

        objects = columns[-2::-1]
        y_pos = np.arange(len(objects))  # Get input feature names
        performance = fi[0, -1::-1]

        plt.barh(y_pos, performance, align='center', alpha=0.5)  # Bar chart
        plt.yticks(y_pos, objects)  # Plot feature names on y-axis
        plt.xlabel('weight')  # x-label
        plt.title('PP (feature importance)')  # Figure heading

        plt.show()  # Display the feature importance

    @staticmethod
    def show_ted_explainer(model, x, y, x_test):
        ted = TED_CartesianExplainer(model)
        ted.fit(x, y)  # train classifier

        # correct answers:  Y:-10; E:13
        y1, e1 = ted.predict_explain(x_test)
        print("Predicting for feature vector:")
        print(" ", x_test[0])
        print("\t\t      Predicted \tCorrect")
        print("Label(Y)\t\t " + np.array2string(y1[0]) + "\t\t   -10")
        print("Explanation (E) \t " + np.array2string(e1[0]) + "\t\t   13")

        ''' X2 = [[3, 1, -11, -2, -2, -2, 296, 0]]

        ## correct answers: Y:-11, E:25
        Y2, E2 = ted.predict_explain(X2)
        print("Predicting for feature vector:")
        print(" ", X2[0])

        print("\t\t      Predicted \tCorrect")
        print("Label(Y)\t\t " + np.array2string(Y2[0]) + "\t\t   -11")
        print("Explanation (E) \t " + np.array2string(E2[0]) + "\t\t   25")'''

    def update_stored_model_file_name_array(self):
        self.storedModelFileNameArray = self.io.get_stored_model_file_name_array()

    def update_stored_eda_file_name_array(self):
        self.storedEdaFileNameArray = self.io.get_stored_eda_file_name_array()

    def check_and_create_related_dirs(self):
        self.io.check_and_create_all_dirs()

    def update_user_scoring_dict(self, user_scoring_dict):
        self.userScoringDict = user_scoring_dict

    def update_scoring_dict(self):
        custom_metrics = CustomMetrics()
        self.scoringDict = custom_metrics.get_scoring_dict(self.type)
        if self.userScoringDict is not None and self.userScoringDict not in self.scoringDict:
            self.scoringDict.update(self.userScoringDict)

    def get_saved_model(self):
        return self.io.get_saved_model()

    def is_model_present(self):
        return self.io.is_model_present()

    def process_test_data_and_predict(self, sub_entity, test, train, label_name, can_drop_label=True,
                                      predict_label_dict=None):

        if not self.is_model_present():
            return

        x_test = self.process_and_get_test_data(test)

        # Predict on testing data:
        model = self.get_saved_model()
        y_pred = model.predict(x_test)

        # RUN EXPLAINABLE AI
        # self.runExplainableAI(model, self.X, self.Y, x_test, self.folderName, self.getTrainColumns(train, labelName))

        if self.type != Variable.typeRegress:
            y_pred = [int(round(value)) for value in y_pred]

        if predict_label_dict is not None:
            y_pred = [predict_label_dict[value] for value in y_pred]
        # answer = {0: 'N', 1: 'Y'}
        # target = [answer[value] for value in y_pred]
        if not can_drop_label:
            pred = []
            for i in y_pred:
                array = []
                for j in i:
                    array.append(float(j))
                pred.append(array)
            y_pred = pred
        print(y_pred[0])
        sub_entity.predictions = y_pred

        self.predict_and_save_predictions(sub_entity)

    def predict_and_save_predictions(self, sub_entity):
        array = self.create_submission_array(sub_entity)
        self.io.write_csv(self.get_submission_file_name(sub_entity.fileName), array, sub_entity.fields)

    def process_and_get_test_data(self, test):
        if test.empty:
            x_test = self.X
        else:
            test_data = self.convert_data(test)
            test_data.fillna(0.0)
            print("COLUMN NULL CHECK TEST")
            print(test_data.isnull().sum(), "\n")
            x_test = np.array(test_data.values.tolist())

        return x_test

    @staticmethod
    def get_submission_file_name(file_name):
        if file_name is None:
            return Variable.submissionFileName
        else:
            return file_name
