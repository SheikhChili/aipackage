# IMPORT
import lightgbm as lgb
from aiautomation.mlpackage.Entities import ClasRegModelEntity
from aiautomation.mlpackage.PackageVariable import Variable
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Models:

    def __init__(self):
        self.alg_name = {}

    # define different models
    # LOGISTIC REGRESSION
    def log_reg(self):
        # define models and parameters˳
        # TODO max_iteration =1000
        logistic_reg = LogisticRegression(max_iter=1100)

        # define grid search
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]

        # define grid search
        grid = dict(solver=solvers, penalty=penalty, C=c_values)

        clas_reg_model_entity = ClasRegModelEntity(logistic_reg, grid, Variable.typeLogReg)
        self.alg_name[logistic_reg] = 'sklearn.linear_model.LogisticRegression'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # RIDGE CLASSIFIER
    def rid_clas(self):
        # define models and parameters
        rid_clas = RidgeClassifier()

        # Whole values
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # define grid search
        grid = dict(alpha=alpha)

        clas_reg_model_entity = ClasRegModelEntity(rid_clas, grid, Variable.typeRidge)
        self.alg_name[rid_clas] = 'sklearn.linear_model.RidgeClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # K NEAREST NEIGHBOUR
    def knn(self):
        # define models and parameters
        knn = KNeighborsClassifier()

        # Whole values
        n_neighbors = list(range(1, 21, 2))
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']

        # define grid search
        grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)

        clas_reg_model_entity = ClasRegModelEntity(knn, grid, Variable.typeKnn)
        self.alg_name[knn] = 'sklearn.neighbors.KNeighborsClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Support Vector Machine (SVM)
    def svm(self):
        # define models and parameters
        svc_classifier = SVC()

        # Whole values
        kernel = ['poly', 'rbf', 'sigmoid']
        c = [100, 10, 1.0, 0.1, 0.01]
        gamma = ['scale', 'auto']

        # define grid search
        grid = dict(kernel=kernel, C=c, gamma=gamma)

        clas_reg_model_entity = ClasRegModelEntity(svc_classifier, grid, Variable.typeSvc)
        self.alg_name[svc_classifier] = 'sklearn.svm.SVC'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # DECISION TREE
    def des_tree(self, max_feature):
        # define models and parameters
        des_tree_classifier = DecisionTreeClassifier()

        # Whole values
        max_depth = [3, None]
        max_features = list(range(1, max_feature, 1))
        min_sample_leaf = list(range(1, 10, 1))
        criterion = ['gini', 'entropy']

        # define grid search
        grid = dict(max_depth=max_depth, max_features=max_features, min_samples_leaf=min_sample_leaf,
                    criterion=criterion)

        clas_reg_model_entity = ClasRegModelEntity(des_tree_classifier, grid, Variable.typeDesTree)
        self.alg_name[des_tree_classifier] = 'sklearn.tree.DecisionTreeClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # RANDOM FOREST
    def ran_for(self):
        # define models and parameters
        ran_for_classifier = RandomForestClassifier()

        # Whole values
        n_estimators = list(range(100, 1001, 100))
        max_features = ['sqrt', 'log2']

        # define grid search
        grid = dict(n_estimators=n_estimators, max_features=max_features)

        clas_reg_model_entity = ClasRegModelEntity(ran_for_classifier, grid, Variable.typeRanFor)
        self.alg_name[ran_for_classifier] = 'sklearn.ensemble.RandomForestClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # GAUSSIAN NB
    def gaussian_nb(self):
        # define models and parameters
        gaussian_nb = GaussianNB()
        var_smoothing = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]

        # define grid search
        grid = dict(var_smoothing=var_smoothing)

        clas_reg_model_entity = ClasRegModelEntity(gaussian_nb, grid, Variable.typeGaussianNB)
        self.alg_name[gaussian_nb] = 'sklearn.naive_bayes.GaussianNB'
        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # BERNOULLI NB
    def bernoulli_nb(self):
        # define models and parameters
        bernoulli_nb = BernoulliNB()
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fit_prior = [False, True]

        # define grid search
        grid = dict(alpha=alpha, fit_prior=fit_prior)

        clas_reg_model_entity = ClasRegModelEntity(bernoulli_nb, grid, Variable.typeBerNB)
        self.alg_name[bernoulli_nb] = 'sklearn.naive_bayes.BernoulliNB'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Stochastic Gradient Boosting
    def grad_boost(self):
        # define models and parameters
        gra_boost_class = GradientBoostingClassifier()

        # Whole values
        n_estimators = list(range(100, 1000, 100))
        learning_rate = [0.001, 0.01, 0.1]
        subsample = [0.5, 0.7, 1.0]
        max_depth = [3, 7, 9]

        # define grid search
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)

        clas_reg_model_entity = ClasRegModelEntity(gra_boost_class, grid, Variable.typeGradBoost)
        self.alg_name[gra_boost_class] = 'sklearn.ensemble.GradientBoostingClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Adaptive Boosting
    def ada_boost(self):
        """
        max_depth = [3,None]
        max_features = [1,2,3,4,5,6,7,8,9,10]
        min_sample_leaf = [1,2,3,4,5,6,7,8,9,10]
        criterion = ['gini','entropy']

        des_classifier = DecisionTreeClassifier(max_depth=max_depth,max_features=max_features,
        min_samples_leaf=min_sample_leaf,criterion=criterion) """

        # define models and parameters
        ada_boost_class = AdaBoostClassifier()
        n_estimators = list(range(50, 1001, 50))
        learning_rate = [0.001, 0.01, 0.1]
        algorithm = ['SAMME', 'SAMME.R']

        # define grid search
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, algorithm=algorithm)

        clas_reg_model_entity = ClasRegModelEntity(ada_boost_class, grid, Variable.typeAdaBoost)
        self.alg_name[ada_boost_class] = 'sklearn.ensemble.AdaBoostClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # XGB boost
    def xgb(self):
        # define models and parameters
        xgb_model = XGBClassifier()

        # Whole values
        max_depth = list(range(3, 10, 2))
        min_child_weight = range(1, 6, 2)
        gamma = [float(i / 10.0) for i in range(0, 5)]
        n_estimators = list(range(100, 1000, 100))
        learning_rate = [0.001, 0.01, 0.1]
        subsample = [i / 10.0 for i in range(6, 10)],
        col_sample_by_tree = [i / 10.0 for i in range(6, 10)]
        reg_alpha = [1e-5, 1e-2, 0.1, 1, 100, 0, 0.001, 0.005, 0.01, 0.05]

        # define grid search
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth,
                    min_child_weight=min_child_weight, gamma=gamma, colsample_bytree=col_sample_by_tree,
                    reg_alpha=reg_alpha)

        clas_reg_model_entity = ClasRegModelEntity(xgb_model, grid, Variable.typeXgb)
        self.alg_name[xgb_model] = 'xgboost.XGBClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Light GBM
    def light_gbm(self):
        # define models and parameters
        light_gbm = lgb.LGBMClassifier()

        # Whole values
        num_leaves = list(range(50, 1000, 50))
        learning_rate = [0.001, 0.01, 0.1]
        max_depth = list(range(3, 10, 2))
        max_bin = list(range(100, 1000, 100))
        objective = 'binary'

        # define grid search
        grid = dict(learning_rate=learning_rate, num_leaves=num_leaves, max_bin=max_bin, max_depth=max_depth,
                    objective=objective)

        clas_reg_model_entity = ClasRegModelEntity(light_gbm, grid, Variable.typeLGBM)
        self.alg_name[light_gbm] = 'lgb.LGBMClassifier'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Cat Boost GBM
    @staticmethod
    def cat_boost_gbm():
        # define models and parameters
        cat_boost = CatBoostClassifier()

        # Whole values
        depth = [3, 1, 2, 6, 4, 5, 7, 8, 9, 10]
        iterations = [250, 100, 500, 1000]
        learning_rate = [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
        l2_leaf_reg = [3, 1, 5, 10, 100]
        border_count = [32, 5, 10, 20, 50, 100, 200],
        ctr_border_count = [50, 5, 10, 20, 100, 200],
        thread_count = 4

        # define grid search
        grid = dict(learning_rate=learning_rate, depth=depth, iterations=iterations, l2_leaf_reg=l2_leaf_reg,
                    border_count=border_count, ctr_border_count=ctr_border_count, thread_count=thread_count)

        clas_reg_model_entity = ClasRegModelEntity(cat_boost, grid, Variable.typeCGBM)
        # self.alg_name[catboost] = 'catboost.CatBoostClassifier'

        return clas_reg_model_entity

    def get_all_models(self, max_feature):
        print(max_feature)
        return [self.rid_clas(), self.log_reg(), self.knn(), self.bernoulli_nb(),
                self.gaussian_nb()]  # ,self.des_tree(max_feature), self.svm(), self.ran_for(), self.ada_boost(),
        # self.grad_boost(), self.xgb(), self.light_gbm(), self.cat_boost_gbm()]

    def get_algorithm_name(self):
        return self.alg_name
