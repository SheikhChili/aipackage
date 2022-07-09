# STORING THE DATA


# IMPORT
import csv
import json
import os.path
from os import path

import pandas as pd
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.models.attention import AttentionLayer
from tensorflow.keras.models import load_model


class InputOutputStream:

    # FUNCTIONS
    # store data in file
    @staticmethod
    def store_dict(filename, data):
        filehandler = open(filename, 'w')
        json.dump(data, filehandler)
        filehandler.close()

    # read data from file
    @staticmethod
    def read_dict(filename):
        with open(filename, 'r') as fileHandle:
            data = fileHandle.read()
            data = json.loads(data)
            fileHandle.close()
            return data

    @staticmethod
    def store_preprocess_array(filename, data):
        temp_df = pd.DataFrame([data])
        df = temp_df.T
        df.columns = df.columns.astype(str)
        df.to_feather(filename)

    @staticmethod
    def get_preprocess_array(filename):
        df = pd.read_feather(filename)
        df.to_csv("Train.csv")
        print(df.values.ravel().tolist())
        return pd.read_feather(filename).values.ravel().tolist()

    # write the csv files
    # Array with ans and id for submission
    '''def arrayOfAns(self,prediction_, test_id):
            ans_array = []
            #for index in range(len(prediction_)):
            #row=[test_id[index],prediction_[index]]
            #ans_array.append(row)
        #	pass
        return ans_array'''

    @staticmethod
    def write_csv(filename, ans_array, fields):
        # writing to csv file
        with open(filename, 'w') as csvFile:
            # creating a csv writer object
            csv_writer = csv.writer(csvFile)
            # writing the fields
            csv_writer.writerow(fields)
            # writing the data rows
            csv_writer.writerows(ans_array)

    def write_acc_model(self, array, fileName):
        df = self.get_stored_model_data(fileName)
        field = ["Filename", "Best Accuracy", "Val Accuracy"]
        big_array = df.values.tolist()
        big_array.append(array[0])
        df = pd.DataFrame(big_array, columns=field)
        df.to_feather(fileName)

    @staticmethod
    def check_file_exist(filename):
        return path.exists(filename)

    def write_model_result(self, array):
        filename = Variable.modelFileName
        df = self.get_stored_model_data(filename)
        if df.empty:
            open(filename, "w+")
        else:
            pass

        field = ["Filename", "loss", "accuracy", "auc", "f1", "precision", "recall", "val_loss", "val_accuracy",
                 "val_auc", "val_f1", "val_precision", "val_recall"]
        big_array = df.values.tolist()
        big_array.append(array)
        self.write_csv(filename, big_array, field)

    def get_stored_model_file_name_array(self, fileName):
        df = self.get_stored_model_data(fileName)
        if df.empty:
            return []
        else:
            return df["Filename"].values.tolist()

    @staticmethod
    def get_stored_model_data(filename):
        if os.path.isfile(filename):
            return pd.DataFrame(pd.read_feather(filename))
        else:
            return pd.DataFrame()

    @staticmethod
    def save_model(model_name, model, dirname):
        model_filename = dirname + Variable.locationSeparator + model_name + Variable.modelExtension
        model.save(model_filename)

    def save_best_model(self, model_name, model):
        self.remove_all_file(dirname)
        self.save_model(model_name, model, Variable.bestPredictModelFolderName)

    def save_all_model(self, model_name, folderPath, model):
        modelFolderPath = Variable.allPredictModelFolderName + Variable.locationSeparator + folderPath
        self.save_model(model_name, model, modelFolderPath)

    @staticmethod
    def remove_all_file(dirname):
        for filename in os.scandir(dirname):
            os.remove(filename.path)

    def check_and_create_all_dirs(self):
        self.check_and_create_dir(Variable.dataFolderName)
        self.check_and_create_dir(Variable.preprocessFolderName)
        self.check_and_create_dir(Variable.searchDataFolderName)
        self.check_and_create_dir(Variable.ranSearchFolderName)
        self.check_and_create_dir(Variable.baySearchFolderName)
        self.check_and_create_dir(Variable.hpSearchFolderName)
        self.check_and_create_dir(Variable.bestPredictModelFolderName)
        self.check_and_create_dir(Variable.allPredictModelFolderName)

    def check_and_save_as_feather(self, actualFolderPath, csvFileName, featherFileName):
        filename = actualFolderPath + csvFileName
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            featherFolderPath = actualFolderPath + Variable.featherFolderName
            self.check_and_create_dir(featherFolderPath)
            featherFilePath = featherFolderPath + featherFileName
            if os.path.isfile(featherFilePath):
                return
            df.to_feather(featherFilePath)

    @staticmethod
    def check_and_create_dir(directoryName):
        if not os.path.exists(directoryName):
            os.mkdir(directoryName)

    def get_best_model(self):
        model_dir = Variable.bestPredictModelFolderName
        model_filename = model_dir + Variable.locationSeparator + os.listdir(model_dir)[0]
        return self.get_loaded_model(model_filename)

    def get_saved_model(self, modelName):
        model_filename = Variable.allPredictModelFolderName + Variable.locationSeparator + modelName + \
                         Variable.modelExtension
        return self.get_loaded_model(model_filename)

    def get_all_saved_model(self):
        model_filename_array = []
        model_dir = Variable.allPredictModelFolderName
        for dir_name in os.listdir(model_dir):
            search_model_dir = Variable.allPredictModelFolderName + Variable.locationSeparator + dir_name
            for search_dir_name in os.listdir(search_model_dir):

                load_model_dir = search_model_dir + Variable.locationSeparator + search_dir_name
                modelFileName = self.get_model_file_name_if_exists(load_model_dir)
                if modelFileName is None:
                    continue
                model_filename = load_model_dir + Variable.locationSeparator + os.listdir(load_model_dir)[0]
                model_filename_array.append(model_filename)
        return model_filename_array

    @staticmethod
    def get_model_file_name_if_exists(dirName):
        fileList = os.listdir(dirName)
        if len(fileList) != 0:
            return fileList[0]
        else:
            return None

    @staticmethod
    def is_model_present():
        return os.listdir(Variable.bestPredictModelFolderName) != []

    @staticmethod
    def get_loaded_model(modelFileName):
        return load_model(modelFileName, compile=False, custom_objects={'AttentionLayer': AttentionLayer})
