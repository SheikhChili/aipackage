# IMPORT
import lightgbm as lgb
from sklearn.svm import SVC
from xgboost import XGBClassifier
from aiautomation.mlpackage.Entities import MultiClasModelEnitity
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from aiautomation.mlpackage.PackageVariable import Variable
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


class Models:

    def __init__(self):
        self.alg_name = {}

    # define different models
    # LOGISTIC REGRESSION
    def log_reg(self):
        # define models and parameters
        # TODO max_iteration =1000
        logistic_reg = LogisticRegression(max_iter=1100)

        # Whole values
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]

        # define grid search
        grid = dict(estimator__solver=solvers, estimator__penalty=penalty, estimator__C=c_values)

        # multi class classification
        ovo = OneVsOneClassifier(logistic_reg)

        # Start the model to run
        self.alg_name[logistic_reg] = 'sklearn.linear_model.LogisticRegression'

        gene_grid = dict(solver=solvers, penalty=penalty, C=c_values)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeLogReg, gene_grid, logistic_reg)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # RIDGE CLASSIFIER
    def rid_clas(self):
        # define models and parameters
        rid_clas = RidgeClassifier()

        # Whole values
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # define grid search
        grid = dict(estimator__alpha=alpha)

        # multi class classification
        ovo = OneVsOneClassifier(rid_clas)

        # Start the model to run
        self.alg_name[rid_clas] = 'sklearn.linear_model.RidgeClassifier'

        gene_grid = dict(alpha=alpha)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeRidge, gene_grid, rid_clas)
        return multi_class_model_entity

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
        grid = dict(estimator__n_neighbors=n_neighbors, estimator__weights=weights, estimator__metric=metric)

        # multi class classification
        ovo = OneVsOneClassifier(knn)

        # Start the model to run
        self.alg_name[knn] = 'sklearn.neighbors.KNeighborsClassifier'

        gene_grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeKnn, gene_grid, knn)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # Support Vector Machine (SVM)
    def svm(self):
        # define models and parameters
        svc_classifier = SVC()

        # Whole values
        kernel = ['poly', 'rbf', 'sigmoid']
        c = [100, 10, 1.0, 0.1, 0.01]
        gamma = ['scale']

        # define grid search
        grid = dict(estimator__kernel=kernel, estimator__C=c, estimator__gamma=gamma)

        # multi class classification
        ovo = OneVsOneClassifier(svc_classifier)

        # Start the model to run	
        self.alg_name[svc_classifier] = 'sklearn.svm.SVC'

        gene_grid = dict(kernel=kernel, C=c, gamma=gamma)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeSvc, gene_grid, svc_classifier)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # DECISION TREE
    def des_tree(self):
        # define models and parameters
        des_tree_classifier = DecisionTreeClassifier()

        # Whole values
        max_depth = [3, None]
        max_features = list(range(1, 10, 1))
        min_sample_leaf = list(range(1, 10, 1))
        criterion = ['gini', 'entropy']

        # define grid search
        grid = dict(estimator__max_depth=max_depth, estimator__max_features=max_features,
                    estimator__min_samples_leaf=min_sample_leaf, estimator__criterion=criterion)

        # multi class classification
        ovo = OneVsOneClassifier(des_tree_classifier)

        # Start the model to run
        self.alg_name[des_tree_classifier] = 'sklearn.tree.DecisionTreeClassifier'

        gene_grid = dict(max_depth=max_depth, max_features=max_features, min_samples_leaf=min_sample_leaf,
                         criterion=criterion)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeDesTree, gene_grid,
                                                         des_tree_classifier)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # RANDOM FOREST
    def ran_for(self):
        # define models and parameters
        ran_for_classifier = RandomForestClassifier()

        # Whole values
        n_estimators = list(range(100, 1001, 100))
        max_features = ['sqrt', 'log2']

        # define grid search
        grid = dict(estimator__n_estimators=n_estimators, estimator__max_features=max_features)

        # multi class classification
        ovo = OneVsOneClassifier(ran_for_classifier)

        # Start the model to runs
        self.alg_name[ran_for_classifier] = 'sklearn.ensemble.RandomForestClassifier'

        gene_grid = dict(n_estimators=n_estimators, max_features=max_features)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeRanFor, gene_grid, ran_for_classifier)
        return multi_class_model_entity

    '''#-----------------------------------------------------------------------------
    #GAUSSIAN NB
    def gaussianNB():
        start = time.time()
        print ("GAUSSIAN NAIVE BAYES START  ------------")

        ## define models and parameters
        gaussian_nb = GaussianNB()
        var_smoothing = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1]

        #define grid search
        grid = dict(var_smoothing = var_smoothing)

         #Start the model alg_name[gaussian_nb] = 'sklearn.naive_bayes.GaussianNB' t11 = threading.Thread(
         target=model_grid_fit,args=(gaussian_nb, X, Y,X_test,grid,"GAUSSIAN_NB_submission_grid")) t12 = 
         threading.Thread(target=model_bayes_fit,args = (gaussian_nb, X, Y,X_test,grid, 
         "GAUSSIAN_NB_submission_bayes")) t13 = threading.Thread(target=model_gene_fit,args = (gaussian_nb, X, Y,
         X_test,grid, "GAUSSIAN_NB_submission_gene")) t14 = threading.Thread(target=model_optuna_fit,
         args = (gaussian_nb, X, Y, X_test,grid,"GAUSSIAN_NB_submission_optuna")) t11.start() t12.start() t13.start() 
         t14.start() t11.join() t12.join() t13.join() t14.join() 

        end = time.time()	
        print ("Gaussian Naive bayes Finish. In time of seconds = ",int(end-start),"\n\n\n")


    #-----------------------------------------------------------------------------
    #MULTINOMIAL NB
    def multinomialNB():
        start = time.time()
        print ("Multinomial NAIVE BAYES START  ------------")

        ## define models and parameters
        multinomial_nb = MultinomialNB()
        alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        fit_prior = [False, True]

        # define grid search
        grid = dict(alpha = alpha, fit_prior = fit_prior)

         #Start the model alg_name[multinomial_nb] = 'sklearn.naive_bayes.MultinomialNB' t11 = threading.Thread(
         target=model_grid_fit,args=(multinomial_nb, X, Y,X_test,grid,"MULTINOMIAL_NB_submission_grid")) t12 = 
         threading.Thread(target=model_bayes_fit,args = (multinomial_nb, X, Y,X_test,grid,
         "MULTINOMIAL_NB_submission_bayes")) t13 = threading.Thread(target=model_gene_fit,args = (multinomial_nb, X, Y,
         X_test,grid,"MULTINOMIAL_NB_submission_gene")) t14 = threading.Thread(target=model_optuna_fit,
         args = (multinomial_nb, X, Y, X_test,grid,"MULTINOMIAL_NB_submission_optuna")) t11.start() t12.start() 
         t13.start() t14.start() t11.join() t12.join() t13.join() t14.join() 

        end = time.time()	
        print ("Multinomial Naive bayes Finish. In time of seconds = ",int(end-start),"\n\n\n")'''

    # -----------------------------------------------------------------------------
    # Stochastic Gradient Boosting
    def grad_boost(self):
        # define models and parameters
        grad_boost_class = GradientBoostingClassifier()

        # Whole values
        n_estimators = list(range(100, 1000, 100))
        learning_rate = [0.001, 0.01, 0.1]
        subsample = [0.5, 0.7, 1.0]
        max_depth = [3, 7, 9]

        # define grid search
        grid = dict(estimator__learning_rate=learning_rate, estimator__n_estimators=n_estimators,
                    estimator__subsample=subsample, estimator__max_depth=max_depth)

        # multi class classification
        ovo = OneVsOneClassifier(grad_boost_class)

        # Start the model
        self.alg_name[grad_boost_class] = 'sklearn.ensemble.GradientBoostingClassifier'

        gene_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                         max_depth=max_depth)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeGradBoost, gene_grid, grad_boost_class)
        return multi_class_model_entity

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
        grid = dict(estimator__learning_rate=learning_rate, estimator__n_estimators=n_estimators,
                    estimator__algorithm=algorithm)

        # multi class classification
        ovo = OneVsOneClassifier(ada_boost_class)

        # Start the model
        self.alg_name[ada_boost_class] = 'sklearn.ensemble.AdaBoostClassifier'

        # define grid search
        gene_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, algorithm=algorithm)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeAdaBoost, gene_grid, ada_boost_class)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # XGB boost
    def xgb(self):
        # define models and parameters
        xgb_model = XGBClassifier()

        # Whole values
        max_depth = list(range(3, 10, 2))
        min_child_weight = list(range(1, 6, 2))
        gamma = [float(i / 10.0) for i in range(0, 5)]
        n_estimators = list(range(100, 1000, 100))
        learning_rate = [0.001, 0.01, 0.1]
        subsample = [i / 10.0 for i in range(6, 10)],
        col_sample_by_tree = [i / 10.0 for i in range(6, 10)]
        reg_alpha = [1e-5, 1e-2, 0.1, 1, 100, 0, 0.001, 0.005, 0.01, 0.05]

        # define grid search
        grid = dict(estimator__learning_rate=learning_rate, estimator__n_estimators=n_estimators,
                    estimator__subsample=subsample, estimator__max_depth=max_depth,
                    estimator__min_child_weight=min_child_weight, estimator__gamma=gamma,
                    estimator__colsample_bytree=col_sample_by_tree, estimator__reg_alpha=reg_alpha)

        # multi class classification
        ovo = OneVsOneClassifier(xgb_model)

        # Start the model
        self.alg_name[xgb_model] = 'xgboost.XGBClassifier'

        gene_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                         max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma,
                         colsample_bytree=col_sample_by_tree, reg_alpha=reg_alpha)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeXgb, gene_grid, xgb_model)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # Light GBM
    def light_gbm(self):
        # define models and parameters
        lightgbm = lgb.LGBMClassifier()

        # Whole values
        num_leaves = list(range(50, 1000, 50))
        learning_rate = [0.001, 0.01, 0.1]
        max_depth = list(range(3, 10, 2))
        max_bin = list(range(100, 1000, 100))
        objective = 'binary'

        # define grid search
        grid = dict(estimator__learning_rate=learning_rate, estimator__num_leaves=num_leaves,
                    estimator__max_bin=max_bin, estimator__max_depth=max_depth, estimator__objective=objective)

        # multi class classification
        ovo = OneVsOneClassifier(lightgbm)

        # Start the model
        self.alg_name[lightgbm] = 'lgb.LGBMClassifier'

        gene_grid = dict(learning_rate=learning_rate, num_leaves=num_leaves, max_bin=max_bin, max_depth=max_depth,
                         objective=objective)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeLGBM, gene_grid, lightgbm)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # Cat Boost GBM
    @staticmethod
    def cat_boost_gbm():
        # define models and parameters
        catboost = CatBoostClassifier()

        # Whole values
        depth = [3, 1, 2, 6, 4, 5, 7, 8, 9, 10]
        iterations = [250, 100, 500, 1000]
        learning_rate = [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
        l2_leaf_reg = [3, 1, 5, 10, 100]
        border_count = [32, 5, 10, 20, 50, 100, 200],
        ctr_border_count = [50, 5, 10, 20, 100, 200],
        thread_count = 4

        # define grid search
        grid = dict(estimator__learning_rate=learning_rate, estimator__depth=depth, estimator__iterations=iterations,
                    estimator__l2_leaf_reg=l2_leaf_reg, estimator__border_count=border_count,
                    estimator__ctr_border_count=ctr_border_count, estimator__thread_count=thread_count)

        # multi class classification
        ovo = OneVsOneClassifier(catboost)

        # Start the model
        # self.alg_name[catboost] = 'catboost.CatBoostClassifier'

        gene_grid = dict(learning_rate=learning_rate, depth=depth, iterations=iterations, l2_leaf_reg=l2_leaf_reg,
                         border_count=border_count, ctr_border_count=ctr_border_count, thread_count=thread_count)

        multi_class_model_entity = MultiClasModelEnitity(ovo, grid, Variable.typeCGBM, gene_grid, catboost)
        return multi_class_model_entity

    def get_all_models(self):
        return [self.rid_clas(), self.log_reg(), self.knn(), self.des_tree(),
                self.svm()]  # , self.ran_for(), self.ada_Boost(), self.grad_boost(), self.xgb(), self.light_gbm(),
        # self.cat_boost_gbm()]

    def get_algorithm_name(self):
        return self.alg_name
