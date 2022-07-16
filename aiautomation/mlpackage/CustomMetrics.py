# IMPORT
from aiautomation.mlpackage.PackageVariable import Variable
from sklearn.metrics import precision_score, make_scorer, accuracy_score, auc, f1_score, jaccard_score, recall_score, \
    roc_auc_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, \
    r2_score, calinski_harabasz_score, davies_bouldin_score, completeness_score, fowlkes_mallows_score, \
    homogeneity_score, rand_score, silhouette_score, v_measure_score, log_loss


class CustomMetrics:

    @staticmethod
    def get_classification_metrics(average='binary'):
        roc_auc = make_scorer(roc_auc_score)
        if average == "weighted":
            roc_auc = make_scorer(roc_auc_score, average=average, multi_class='ovo')
        return {'accuracy': make_scorer(accuracy_score), 'f1_score': make_scorer(f1_score, average=average),
                'precision': make_scorer(precision_score, average=average),
                'recall': make_scorer(recall_score, average=average),
                'jaccard': make_scorer(jaccard_score, average=average), 'neg_log_loss': make_scorer(log_loss),
                'roc_auc': roc_auc}

    @staticmethod
    def get_regression_metrics():
        return {'mae': make_scorer(mean_absolute_error), "mse": make_scorer(mean_squared_error),
                "mape": make_scorer(mean_absolute_percentage_error), "r2": make_scorer(
                r2_score)}  # , "msle":make_scorer(mean_squared_log_error)} #TODO:msle - not working if it is
        # negative values

    @staticmethod
    def get_segmentation_metrics():
        return {'accuracy': make_scorer(accuracy_score), "completeness_score": make_scorer(completeness_score),
                "fowlkes_mallows_score": make_scorer(fowlkes_mallows_score),
                "homogeneity_score": make_scorer(homogeneity_score), "rand_score": make_scorer(rand_score),
                "v_measure_score": make_scorer(v_measure_score), "silhouette_score": make_scorer(silhouette_score),
                'calinski_harabasz_score': make_scorer(calinski_harabasz_score),
                "davies_bouldin_score": make_scorer(davies_bouldin_score)}

    def get_scoring_dict(self, type_value):
        if type_value == Variable.typeMultiClass:
            scoring_dict = self.get_classification_metrics("weighted")
        elif type_value == Variable.typeClassification:
            scoring_dict = self.get_classification_metrics()
        elif type_value == Variable.typeRegress:
            scoring_dict = self.get_regression_metrics()
        else:
            scoring_dict = self.get_segmentation_metrics()
        return scoring_dict
