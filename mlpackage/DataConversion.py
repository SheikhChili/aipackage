# IMPORT
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler


class DataConversion:

    # Functions used in this project
    @staticmethod
    def convert_to_num(dataset):
        cat_data = []
        num_data = []

        for i, c in enumerate(dataset.dtypes):
            if c == object:
                cat_data.append(dataset.iloc[:, i])
            else:
                num_data.append(dataset.iloc[:, i])

        cat_data = pd.DataFrame(cat_data).transpose()
        num_data = pd.DataFrame(num_data).transpose()

        # cat_data
        # If you want to fill every column with its own most frequent value you can use
        cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

        # num_data
        # fill every missing value with their previous value in the same column
        # num_data.fillna(method='bfill', inplace=True)

        num_data = num_data.apply(lambda x: x.fillna(x.mean()), axis=0)

        # Transform categorical data
        le = LabelEncoder()
        for i in cat_data:
            cat_data[i] = le.fit_transform(cat_data[i])

        df = pd.concat([cat_data, num_data], axis=1)
        return df

    @staticmethod
    def normalize_data(x):
        # Normalize numerical data
        # fit scaler on training data
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        return pd.DataFrame(x_scaled)

    @staticmethod
    def create_result_array(acc_entity):
        array = []
        array1 = []

        # storing in array
        array.append(acc_entity.filename)
        array.append(acc_entity.score)
        array.append(acc_entity.val_score)
        array.append(acc_entity.param)
        array.append(acc_entity.metrics)
        array1.append(array)

        return array1

    @staticmethod
    def create_submission_array(sub_entity):

        array1 = []
        y_pred = sub_entity.predictions
        id_ = sub_entity.id_
        id_2 = sub_entity.id_2
        id_3 = sub_entity.id_3

        for i in range(len(y_pred)):
            array = []
            # storing in array
            if id_:
                array.append(id_[i])

            if id_2:
                array.append(id_2[i])

            if id_3:
                array.append(id_3[i])

            array.append(y_pred[i])

            array1.append(array)

        return array1

    @staticmethod
    def classification_synthesize_data(train, df, label_name):
        os = RandomOverSampler(1.0)  # Ratio decides how much percent of majority minority should have
        smote_x_train, smote_y_train = os.fit_resample(train, df)
        print("LABEL COUNT AFTER SMOTE")
        print(smote_y_train.value_counts(), "\n")

        '''
        # Synthesize minority class data_points using SMOTE
        sm = SMOTE(random_state=42, sampling_strategy='minority')
        smote_x_train, smote_y_train = sm.fit_resample(train, df)
        # Separate into training and test sets
        smote_x_train = pd.DataFrame(smote_x_train, columns = train.columns)
        smote_y_train = pd.DataFrame(smote_y_train, columns = ['Loan_Status'])
        print(smote_y_train,"\n")
        '''

        return smote_x_train, smote_y_train

    @staticmethod
    def synthesize_data(train, df, label_name):
        print("LABEL COUNT BEFORE SMOTE")
        print(df.value_counts(), "\n")
        # Synthesize minority class data points using SMOTE
        sm = SMOTE(random_state=42, sampling_strategy='minority')
        smote_x_train, smote_y_train = train, df
        # Separate into training and test sets
        label_count = len(df.value_counts()) - 1
        for i in range(0, label_count):
            smote_x_train, smote_y_train = sm.fit_resample(smote_x_train, smote_y_train)
            smote_x_train = pd.DataFrame(smote_x_train, columns=train.columns)
            smote_y_train = pd.DataFrame(smote_y_train, columns=[label_name])

        print("LABEL COUNT AFTER SMOTE")
        print(smote_y_train[label_name].value_counts(), "\n")

        '''
        #Randomoversampler
        os=RandomOverSampler(1.0) #Ratio decides how much percent of majority minority should have
        smote_x_train,smote_y_train=os.fit_sample(train,df)
        # Separate into training and test sets
        smote_x_train = pd.DataFrame(smote_x_train, columns = train.columns)
        smote_y_train = pd.DataFrame(smote_y_train, columns = ['Surge_Pricing_Type'])
        smote_x_train,smote_y_train=os.fit_sample(train,df)
        smote_x_train = pd.DataFrame(smote_x_train, columns = train.columns)
        smote_y_train = pd.DataFrame(smote_y_train, columns = ['Surge_Pricing_Type'])
        print (smote_y_train['Surge_Pricing_Type'].value_counts(),"\n")'''

        return smote_x_train, smote_y_train
