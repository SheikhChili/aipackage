# IMPORT

import sys
import csv
import pickle
import os.path
import pandas as pd
from aipackage.mlpackage.PackageVariable import Variable


class InputOutputStream:

    def __init__(self):
        self.storeDataDirName = ''
        self.update_store_data_dir_name()

    def update_store_data_dir_name(self):
        if Variable.isRunAllFileEnabled:
            self.storeDataDirName = sys.argv[1] + Variable.locationSeparator

    # write the csv files

    @staticmethod
    def write_csv(filename, array, fields):

        # fields = ["Trip_ID","Surge_Pricing_Type"]
        # writing to csv file

        with open(filename, 'w') as csvfile:
            # creating a csv writer object

            csvwriter = csv.writer(csvfile)

            # writing the fields

            csvwriter.writerow(fields)

            # writing the data rows

            csvwriter.writerows(array)

    def write_acc_model(self, array):
        filename = self.storeDataDirName + Variable.modelFileName
        field = ['Filename', 'Best_Score', 'Best_Val_Score', 'Best_Hyper_Parameters', 'Metrics_Name']
        df = self.get_stored_model_data(filename)
        big_array = df.values.tolist()
        big_array.append(array[0])
        df = pd.DataFrame(big_array, columns=field)
        df.to_feather(filename)

    def get_stored_model_file_name_array(self):
        df = self.get_stored_model_data(self.storeDataDirName + Variable.modelFileName)
        if df.empty:
            return []
        else:
            return df['Filename'].values.tolist()

    @staticmethod
    def get_stored_model_data(filename):
        if os.path.isfile(filename):
            return pd.DataFrame(pd.read_feather(filename))
        else:
            return pd.DataFrame()

    def save_best_model(self, model_name, model):
        self.remove_all_pickle_file()
        self.save_model(model_name, model, self.storeDataDirName + Variable.bestPickleFolderName)

    def save_all_model(self, model_name, folderPath, model):
        pickleFolderPath = self.storeDataDirName + Variable.allPickleFolderName + Variable.locationSeparator \
                           + folderPath
        self.save_model(model_name, model, pickleFolderPath)

    def save_all_visualizer(self, model_name, folderPath, model, ):
        visualizerFolderPath = self.storeDataDirName + Variable.allVisualizerFolderName + Variable.locationSeparator \
                               + folderPath
        self.save_model(model_name, model, visualizerFolderPath)

    @staticmethod
    def save_model(model_name, model, pickleFolderPath):
        pickle_filename = pickleFolderPath + Variable.locationSeparator + model_name + Variable.pickleExtension
        pickle.dump(model, open(pickle_filename, Variable.writeBinary))

    def export_gene(self, tpot, fileName):
        gene_file_name = self.storeDataDirName + Variable.geneFolderName + Variable.locationSeparator + fileName \
                         + Variable.pythonExtension
        tpot.export(gene_file_name)

    def remove_all_pickle_file(self):
        for filename in os.scandir(self.storeDataDirName + Variable.bestPickleFolderName):
            os.remove(filename.path)

    def check_and_create_all_dirs(self):
        self.check_and_create_dir(self.storeDataDirName + Variable.dataFolderName)
        self.check_and_create_dir(self.storeDataDirName + Variable.geneFolderName)
        self.check_and_create_dir(self.storeDataDirName + Variable.allPickleFolderName)
        self.check_and_create_dir(self.storeDataDirName + Variable.bestPickleFolderName)
        self.check_and_create_dir(self.storeDataDirName + Variable.allVisualizerFolderName)
        self.check_and_create_dir(self.storeDataDirName + Variable.edaLocation)

    def create_model_folder(self, folderPath):
        self.create_pickle_folder(folderPath)
        self.create_visualizer_folder(folderPath)

    def create_visualizer_folder(self, folder_path):
        visualizerFolderPath = self.storeDataDirName + Variable.allVisualizerFolderName + Variable.locationSeparator \
                               + folder_path
        self.check_and_create_dir(visualizerFolderPath)

    def create_pickle_folder(self, folder_path):
        pickleFolderPath = self.storeDataDirName + Variable.allPickleFolderName + Variable.locationSeparator \
                           + folder_path
        self.check_and_create_dir(pickleFolderPath)

    @staticmethod
    def check_and_create_dir(directoryName):
        if not os.path.exists(directoryName):
            os.mkdir(directoryName)

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

    def get_saved_model(self):
        pickle_model_dir = self.storeDataDirName + Variable.bestPickleFolderName
        pickle_filename = pickle_model_dir + Variable.locationSeparator + os.listdir(pickle_model_dir)[0]
        print(pickle_filename)
        pickle_file = open(pickle_filename, Variable.readBinary)
        return pickle.load(pickle_file)

    def is_model_present(self):
        return os.listdir(self.storeDataDirName + Variable.bestPickleFolderName) != []