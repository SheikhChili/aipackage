import math
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import AUC, RootMeanSquaredError
from aipackage.dlpackage.PackageVariable import Variable
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


class CustomMetrics:

    @staticmethod
    def recall_m(y_true, y_pred):
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + k.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + k.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + k.epsilon()))

    def get_classification_metrics(self):
        # with Variable.distribution_strategy.scope():
        auc = AUC()
        root_mean_sq_error = RootMeanSquaredError()

        return ['accuracy', auc, root_mean_sq_error, self.f1_m, self.precision_m, self.recall_m]

    @staticmethod
    def get_regression_metrics():
        # with Variable.distribution_strategy.scope():

        return ["mean_squared_error", "root_mean_squared_error", "mean_absolute_error",
                "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "cosine_similarity", "logcosh"]

    def get_nlp_metrics(self, nlp_type):
        if nlp_type == Variable.typeRegression:
            return self.get_regression_metrics()
        else:
            return self.get_classification_metrics()


class CustomEarlyStopping(Callback):
    def __init__(self, loss_monitor='loss', loss_value=0.5, acc_monitor="accuracy", acc_value=0.99, val_loss_value=0.9,
                 val_loss_monitor='val_loss', verbose=1):
        super(Callback, self).__init__()
        self.loss_monitor = loss_monitor
        self.loss_value = loss_value
        self.acc_monitor = acc_monitor
        self.acc_value = acc_value
        self.val_loss_monitor = val_loss_monitor
        self.val_loss_value = val_loss_value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        temp_loss_value = logs.get(self.loss_monitor)
        temp_acc_value = logs.get(self.acc_monitor)
        temp_val_loss_value = logs.get(self.val_loss_monitor)
        if temp_loss_value is None or temp_acc_value is None:
            warnings.warn("Early stopping requires %s available!", RuntimeWarning)
            assert (0 == 1)
        print("Epoch %05d: early stopping THR" % epoch)
        print("TEMP ACC VALUE = ", temp_acc_value)
        print("TEMP LOSS VALUE = ", temp_loss_value)
        print("TEMP VAL LOSS VALUE = ", temp_val_loss_value)
        if temp_acc_value > self.acc_value:
            if temp_loss_value < self.loss_value:  # and temp_val_loss_value < self.val_loss_value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR STOP CALLED" % epoch)
                    print("LOGS = ", logs)
                self.model.stop_training = True


class CustomLearningRateScheduler:
    @staticmethod
    def step_decay(epoch):
        initial_learning_rate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        learning_rate = initial_learning_rate * math.pow(drop, math.floor(epoch / epochs_drop))
        print(learning_rate)
        return learning_rate

    @staticmethod
    def lr_scheduler(epoch, lr):
        decay_rate = 0.1
        decay_step = 90
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    '''def step_decay(self, losses):
        if float(2*np.sqrt(np.array(history.losses[-1])))<0.3:
            learning_rate=0.01*1/(1+0.1*len(history.losses))
            momentum=0.8
            decay_rate=2e-6
            return learning_rate
        else:
            learning_rate=0.1
            return learning_rate'''

    @staticmethod
    def step_exp_decay(epoch):
        cur_epoch = epoch + 1
        learning_rate = 3e-4
        return (learning_rate / cur_epoch).astype('float32')

    '''def getLearningRateScheduler(self): return CyclicLR(base_lr = Variable.base_lr, max_lr = Variable.max_lr, 
    step_size = Variable.step_size, mode = Variable.exp_mode, gamma=Variable.gamma) '''

    @staticmethod
    def get_learning_rate_scheduler():
        return ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=3e-4)

    '''def getLearningRateScheduler(self):
           return LearningRateScheduler(self.step_decay)'''
