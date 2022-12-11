# IMPORT
import lightgbm as lgb
from sklearn.svm import SVC
from xgboost import XGBClassifier
from aiautomation.mlpackage.Entities import MultiClasModelEntity, MultiClassGridEntity
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

        # multi class classification
        ovo = OneVsOneClassifier(logistic_reg)

        # Start the model to run
        self.alg_name[logistic_reg] = 'sklearn.linear_model.LogisticRegression'
        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeLogReg)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeLogReg,
                                                        multi_class_grid_entity.gene_grid, logistic_reg)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # RIDGE CLASSIFIER
    def rid_clas(self):
        # define models and parameters
        rid_clas = RidgeClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(rid_clas)

        # Start the model to run
        self.alg_name[rid_clas] = 'sklearn.linear_model.RidgeClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeRidge)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeRidge,
                                                        multi_class_grid_entity.gene_grid, rid_clas)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # K NEAREST NEIGHBOUR
    def knn(self):
        # define models and parameters
        knn = KNeighborsClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(knn)

        # Start the model to run
        self.alg_name[knn] = 'sklearn.neighbors.KNeighborsClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeKnn)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeKnn,
                                                        multi_class_grid_entity.gene_grid, knn)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # Support Vector Machine (SVM)
    def svm(self):
        # define models and parameters
        svc_classifier = SVC()

        # multi class classification
        ovo = OneVsOneClassifier(svc_classifier)

        # Start the model to run	
        self.alg_name[svc_classifier] = 'sklearn.svm.SVC'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeSvc)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeSvc,
                                                        multi_class_grid_entity.gene_grid, svc_classifier)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # DECISION TREE
    def des_tree(self):
        # define models and parameters
        des_tree_classifier = DecisionTreeClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(des_tree_classifier)

        # Start the model to run
        self.alg_name[des_tree_classifier] = 'sklearn.tree.DecisionTreeClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeDesTree)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeDesTree,
                                                        multi_class_grid_entity.gene_grid,
                                                        des_tree_classifier)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # RANDOM FOREST
    def ran_for(self):
        # define models and parameters
        ran_for_classifier = RandomForestClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(ran_for_classifier)

        # Start the model to runs
        self.alg_name[ran_for_classifier] = 'sklearn.ensemble.RandomForestClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeRanFor)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeRanFor,
                                                        multi_class_grid_entity.gene_grid, ran_for_classifier)
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

        # multi class classification
        ovo = OneVsOneClassifier(grad_boost_class)

        # Start the model
        self.alg_name[grad_boost_class] = 'sklearn.ensemble.GradientBoostingClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeGradBoost)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeGradBoost,
                                                        multi_class_grid_entity.gene_grid, grad_boost_class)
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
        ada_boost_class = AdaBoostClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(ada_boost_class)

        # Start the model
        self.alg_name[ada_boost_class] = 'sklearn.ensemble.AdaBoostClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeAdaBoost)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeAdaBoost,
                                                        multi_class_grid_entity.gene_grid, ada_boost_class)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # XGB boost
    def xgb(self):
        # define models and parameters
        xgb_model = XGBClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(xgb_model)

        # Start the model
        self.alg_name[xgb_model] = 'xgboost.XGBClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeXGB)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeXGB,
                                                        multi_class_grid_entity.gene_grid, xgb_model)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # Light GBM
    def light_gbm(self):
        # define models and parameters
        lightgbm = lgb.LGBMClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(lightgbm)

        # Start the model
        self.alg_name[lightgbm] = 'lgb.LGBMClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeLGBM)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeLGBM,
                                                        multi_class_grid_entity.gene_grid, lightgbm)
        return multi_class_model_entity

    # -----------------------------------------------------------------------------
    # Cat Boost GBM
    def cat_boost_gbm(self):
        # define models and parameters
        catboost = CatBoostClassifier()

        # multi class classification
        ovo = OneVsOneClassifier(catboost)

        # Start the model
        # self.alg_name[catboost] = 'catboost.CatBoostClassifier'

        multi_class_grid_entity = self.check_and_get_grid_entity(Variable.typeCGBM)
        multi_class_model_entity = MultiClasModelEntity(ovo, multi_class_grid_entity.grid, Variable.typeCGBM,
                                                        multi_class_grid_entity.gene_grid, catboost)
        return multi_class_model_entity

    def get_all_models(self):
        return [self.rid_clas(), self.log_reg(), self.knn(), self.des_tree(),
                self.svm()]  # , self.ran_for(), self.ada_Boost(), self.grad_boost(), self.xgb(), self.light_gbm(),
        # self.cat_boost_gbm()]

    def get_algorithm_name(self):
        return self.alg_name

    def check_and_get_grid(self, model_type, search_type):
        multi_class_grid_entity = self.check_and_get_grid_entity(model_type)
        grid = multi_class_grid_entity.grid
        if search_type == Variable.typeGene:
            grid = multi_class_grid_entity.gene_grid
        return grid

    @staticmethod
    def check_and_get_grid_entity(model_type):
        multi_class_grid_entity = None
        if model_type == Variable.typeRidge:
            # Whole values
            alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(dict(estimator__alpha=alpha), dict(alpha=alpha))

        elif model_type == Variable.typeLogReg:
            # Whole values
            solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            penalty = ['l2']
            c_values = [100, 10, 1.0, 0.1, 0.01]

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(
                dict(estimator__solver=solvers, estimator__penalty=penalty, estimator__C=c_values)
                , dict(solver=solvers, penalty=penalty, C=c_values))

        elif model_type == Variable.typeKnn:
            # Whole values
            n_neighbors = list(range(1, 21, 2))
            weights = ['uniform', 'distance']
            metric = ['euclidean', 'manhattan', 'minkowski']

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(dict(estimator__n_neighbors=n_neighbors,
                                                                estimator__weights=weights, estimator__metric=metric),
                                                           dict(n_neighbors=n_neighbors, weights=weights,
                                                                metric=metric))

        elif model_type == Variable.typeSvc:
            # Whole values
            kernel = ['poly', 'rbf', 'sigmoid']
            c = [100, 10, 1.0, 0.1, 0.01]
            gamma = ['scale']

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(dict(estimator__kernel=kernel, estimator__C=c,
                                                                estimator__gamma=gamma), dict(kernel=kernel, C=c,
                                                                                              gamma=gamma))

        elif model_type == Variable.typeDesTree:
            # Whole values
            # Whole values
            max_depth = [3, None]
            max_features = list(range(1, 10, 1))
            min_sample_leaf = list(range(1, 10, 1))
            criterion = ['gini', 'entropy']

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(dict(estimator__max_depth=max_depth,
                                                                estimator__max_features=max_features,
                                                                estimator__min_samples_leaf=min_sample_leaf,
                                                                estimator__criterion=criterion),
                                                           dict(max_depth=max_depth, max_features=max_features,
                                                                min_samples_leaf=min_sample_leaf,
                                                                criterion=criterion))

        elif model_type == Variable.typeRanFor:
            # Whole values
            n_estimators = list(range(100, 1001, 100))
            max_features = ['sqrt', 'log2']

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(
                dict(estimator__n_estimators=n_estimators, estimator__max_features=max_features),
                dict(n_estimators=n_estimators, max_features=max_features))

        elif model_type == Variable.typeGradBoost:
            # Whole values
            n_estimators = list(range(100, 1000, 100))
            learning_rate = [0.001, 0.01, 0.1]
            subsample = [0.5, 0.7, 1.0]
            max_depth = [3, 7, 9]

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(
                dict(estimator__learning_rate=learning_rate, estimator__n_estimators=n_estimators,
                     estimator__subsample=subsample, estimator__max_depth=max_depth),
                dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                     max_depth=max_depth))

        elif model_type == Variable.typeAdaBoost:
            n_estimators = list(range(50, 1001, 50))
            learning_rate = [0.001, 0.01, 0.1]
            algorithm = ['SAMME', 'SAMME.R']

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(
                dict(estimator__learning_rate=learning_rate, estimator__n_estimators=n_estimators,
                     estimator__algorithm=algorithm),
                dict(learning_rate=learning_rate, n_estimators=n_estimators, algorithm=algorithm))

        elif model_type == Variable.typeXGB:
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
            multi_class_grid_entity = MultiClassGridEntity(
                dict(estimator__learning_rate=learning_rate, estimator__n_estimators=n_estimators,
                     estimator__subsample=subsample, estimator__max_depth=max_depth,
                     estimator__min_child_weight=min_child_weight, estimator__gamma=gamma,
                     estimator__colsample_bytree=col_sample_by_tree, estimator__reg_alpha=reg_alpha),
                dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                     max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma,
                     colsample_bytree=col_sample_by_tree, reg_alpha=reg_alpha))

        elif model_type == Variable.typeLGBM:
            # Whole values
            num_leaves = list(range(50, 1000, 50))
            learning_rate = [0.001, 0.01, 0.1]
            max_depth = list(range(3, 10, 2))
            max_bin = list(range(100, 1000, 100))
            objective = 'binary'

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(
                dict(estimator__learning_rate=learning_rate, estimator__num_leaves=num_leaves,
                     estimator__max_bin=max_bin, estimator__max_depth=max_depth, estimator__objective=objective),
                dict(learning_rate=learning_rate, num_leaves=num_leaves, max_bin=max_bin, max_depth=max_depth,
                     objective=objective))

        elif model_type == Variable.typeCGBM:
            # Whole values
            depth = [3, 1, 2, 6, 4, 5, 7, 8, 9, 10]
            iterations = [250, 100, 500, 1000]
            learning_rate = [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
            l2_leaf_reg = [3, 1, 5, 10, 100]
            border_count = [32, 5, 10, 20, 50, 100, 200],
            ctr_border_count = [50, 5, 10, 20, 100, 200],
            thread_count = 4

            # define grid search
            multi_class_grid_entity = MultiClassGridEntity(
                dict(estimator__learning_rate=learning_rate, estimator__depth=depth,
                     estimator__iterations=iterations,
                     estimator__l2_leaf_reg=l2_leaf_reg, estimator__border_count=border_count,
                     estimator__ctr_border_count=ctr_border_count, estimator__thread_count=thread_count),
                dict(learning_rate=learning_rate, depth=depth, iterations=iterations, l2_leaf_reg=l2_leaf_reg,
                     border_count=border_count, ctr_border_count=ctr_border_count, thread_count=thread_count))

        return multi_class_grid_entity
