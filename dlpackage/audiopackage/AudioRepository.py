# import
import numpy as np
import threading
import librosa
from aipackage.dlpackage.StoreData import InputOutputStream
from aipackage.dlpackage.annpackage.AnnRepository import AnnRepository
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from aipackage.dlpackage.PackageVariable import Variable
from aipackage.dlpackage.Entities import DatasetEntity, ModelEntity


class AudioRepository(AnnRepository):

    def __init__(self):
        super().__init__()
        super().set_type(Variable.typeAudio)
        self.updateDatasetLocation(Variable.audioDatasetLocation)
        self.oriTrainData = []
        self.oriLabelData = []
        self.oriTestData = []

    def set_ori_train_data(self, train_data):
        self.oriTrainData = train_data

    def set_ori_label_data(self, label_data):
        self.oriLabelData = label_data

    def set_ori_test_data(self, test_data):
        self.oriTestData = test_data

    def get_ori_train_data(self):
        return self.oriTrainData

    def get_ori_label_data(self):
        return self.oriLabelData

    def get_ori_test_data(self):
        return self.oriTestData

    def check_or_create_required_data_exist(self, ori_train_data, label_data, ori_test_data=None):
        if ori_test_data is None:
            ori_test_data = []
        if super().is_data_present():
            super().set_all_train_data()
        else:
            self.preprocess_data(ori_train_data, label_data, ori_test_data)
            self.check_and_store_data()

    def check_and_store_data(self):
        if not super().get_ori_preprocess_train_sentence_array():
            return
        fileNamePrefix = Variable.preprocessFolderName + Variable.locationSeparator
        ioStream = InputOutputStream()
        super().store_data(fileNamePrefix, ioStream)
        t1 = threading.Thread(target=ioStream.store_preprocess_array,
                              args=(fileNamePrefix + Variable.labelFileName, super().get_ori_label_array()))
        t1.start()
        t1.join()

    def preprocess(self, ori_train_data, label_data, ori_test_data):
        self.check_or_create_required_data_exist(ori_train_data, label_data, ori_test_data)

    def preprocess_data(self, ori_train_data, label_data, ori_test_data):
        super().set_ori_preprocess_train_sentence_array(self.preprocess_audio_data(ori_train_data))
        super().set_ori_label_array(label_data)
        super().set_ori_preprocess_test_sentence_array(self.preprocess_audio_data(ori_test_data))

    # setOriPreprocessTestSentenceArray

    def preprocess_audio_data(self, data):
        # data = librosa.resample(data, sample_rate, Variable.sample_rate)
        extracted_feature = []
        i = 0
        for samples in data:
            i += 1
            extracted_feature.append(self.features_extractor(samples))
            if i % 500 == 0:
                print("FINISHED")
        return extracted_feature

    @staticmethod
    def features_extractor(samples):
        mfccs_features = librosa.feature.mfcc(y=samples, sr=Variable.sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features

    def get_xy_data(self):
        return super().get_xy_data(self.get_limited_train_data(), self.get_limited_label_data())

    def get_limited_train_data(self):
        return super().get_ori_preprocess_train_sentence_array()[:super().get_data_limit(Variable.TYPE_EXTRAS_TRAIN)]

    def get_limited_label_data(self):
        return super().get_ori_label_array()[:super().get_data_limit(Variable.TYPE_EXTRAS_TRAIN)]

    '''def getDataLimit(self, data_type):
        if data_type == Variable.TYPE_EXTRAS_TEST:
            return self.getTestDataLimit()
        else:
            return self.getTrainDataLimit()
            
                    
    def getTrainDataLimit(self):
        trainDataLimit = Variable.trainDataLimit
        if super().getTrainAllData():
            trainDataLimit = len(self.getOriTrainData())
        return trainDataLimit
        
        
    def getTestDataLimit(self):
        testDataLimit = Variable.testDataLimit
        if super().getTestAllData():
            testDataLimit = len(self.getOriTestData())
        return testDataLimit'''

    def update_user_metrics(self, user_metrics=None):
        super().update_user_metrics(user_metrics)

    def start_train(self):
        super().create_and_run_model(self.get_entity())

    def get_entity(self):
        array = self.get_xy_data()
        X = array[0]
        y = array[1]

        print("\nUNIQUE COUNTS = ", np.unique(y, return_counts=True), "\n\n")
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)

        # convert integers to dummy variables (i.e. one hot encoded)
        Y = to_categorical(encoded_Y)
        print("TRAIN SENTENCE SHAPE = ", X.shape)
        print("LABEL = ", Y.shape, "\n")

        return self.get_model_entity(Y.shape[1]), DatasetEntity(X, Y)

    def get_model_entity(self, target_size):
        final_activation = 'softmax'
        loss = 'categorical_crossentropy'
        optimizer = 'adam'
        metrics = super().get_metrics()
        return ModelEntity(target_size, final_activation, loss, optimizer, metrics)
