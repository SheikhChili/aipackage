# IMPORT
import numpy as np
import lightgbm as lgb
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from aiautomation.mlpackage.Entities import ClasRegModelEntity
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from aiautomation.mlpackage.PackageVariable import Variable
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


class Models:

    def __init__(self):
        self.alg_name = {}

    # define different models
    # LOGISTIC REGRESSION
    def lin_reg(self):
        # define models and parameters
        linreg = LinearRegression()
        fit_intercept = [True, False]
        # normalize = [True, False]
        copy_x = [True, False]

        # define grid search
        grid = dict(fit_intercept=fit_intercept, copy_X=copy_x)

        clas_reg_model_entity = ClasRegModelEntity(linreg, grid, Variable.typeLinReg)
        self.alg_name[linreg] = 'sklearn.linear_model.LinearRegression'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # RIDGE CLASSIFIER
    def rid_reg(self):
        # define models and parameters
        rid_reg = Ridge()
        alpha = [0.0001, 0.001, 0.01, 0.1, 1.0]
        # normalize = [True, False]
        fit_intercept = [True, False]
        solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

        # define grid search
        grid = dict(alpha=alpha, solver=solver, fit_intercept=fit_intercept)

        clas_reg_model_entity = ClasRegModelEntity(rid_reg, grid, Variable.typeRidge)
        self.alg_name[rid_reg] = 'sklearn.linear_model.Ridge'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # K NEAREST NEIGHBOUR
    def knn(self):
        # define models and parameters
        knn = KNeighborsRegressor()
        n_neighbors = list(range(1, 21, 2))
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']

        # define grid search
        grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)

        clas_reg_model_entity = ClasRegModelEntity(knn, grid, Variable.typeKnn)
        self.alg_name[knn] = 'sklearn.neighbors.KNeighborsRegressor'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Support Vector Machine (SVM)
    def svr(self):
        # define models and parameters
        svr = SVR()
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        gamma = ['scale', 'auto']
        c = [100, 10, 1.0, 0.1, 0.01]
        # epsilon = [0.1,0.2,0.5,0.3]

        # define grid search
        grid = dict(kernel=kernel, gamma=gamma, C=c)

        clas_reg_model_entity = ClasRegModelEntity(svr, grid, Variable.typeSvc)
        self.alg_name[svr] = 'sklearn.svm.SVR'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # DECISION TREE
    def des_tree(self):
        # define models and parameters
        des_tree_reg = DecisionTreeRegressor()
        criterion = ['mse', 'friedman_mse', 'mae']
        splitter = ['best', 'random']
        max_features = ['auto', 'sqrt', 'log2']
        max_depth = list(range(5, 150, 15))
        min_samples_leaf = list(range(10, 150, 10))
        min_samples_leaf.append(1)
        max_leaf_nodes = list(range(5, 150, 15))
        min_samples_split = list(range(10, 150, 10))

        # define grid search
        grid = dict(criterion=criterion, splitter=splitter, max_features=max_features, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split)

        clas_reg_model_entity = ClasRegModelEntity(des_tree_reg, grid, Variable.typeDesTree)
        self.alg_name[des_tree_reg] = 'sklearn.tree.DecisionTreeRegressor'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # RANDOM FOREST
    def ran_for(self):
        # define models and parameters
        ran_for_reg = RandomForestRegressor()
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
        max_features = ['auto', 'sqrt', 'log2']
        max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10, 15, 100]
        min_samples_leaf = [1, 2, 5, 10]
        criterion = ['mse', 'mae']

        # define grid search
        grid = dict(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf, criterion=criterion)

        clas_reg_model_entity = ClasRegModelEntity(ran_for_reg, grid, Variable.typeRanFor)
        self.alg_name[ran_for_reg] = 'sklearn.ensemble.RandomForestRegressor'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Stochastic Gradient Boosting
    def grad_boost(self):
        # define models and parameters
        gra_boost_reg = GradientBoostingRegressor()
        loss = ['ls', 'lad', 'huber', 'quantile']
        learning_rate = [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
        n_estimators = list(range(100, 1000, 100))
        criterion = ['friedman_mse', 'mse', 'mae']
        subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # alpha = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        max_features = ['auto', 'sqrt', 'log2']
        max_depth = list(range(1, 11, 1))

        # define grid search
        grid = dict(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion,
                    subsample=subsample,
                    max_features=max_features, max_depth=max_depth)

        clas_reg_model_entity = ClasRegModelEntity(gra_boost_reg, grid, Variable.typeGradBoost)
        self.alg_name[gra_boost_reg] = 'sklearn.ensemble.GradientBoostingRegressor'

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
        ada_boost_reg = AdaBoostRegressor()
        n_estimators = list(range(50, 1001, 50))
        learning_rate = [0.001, 0.01, 0.1]
        loss = ['linear', 'square', 'exponential']

        # define grid search
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, loss=loss)

        clas_reg_model_entity = ClasRegModelEntity(ada_boost_reg, grid, Variable.typeAdaBoost)
        self.alg_name[ada_boost_reg] = 'sklearn.ensemble.AdaBoostRegressor'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # XGB boost
    def xgb(self):
        # define models and parameters
        xgb_model = XGBRegressor()
        max_depth = list(range(3, 10, 2))
        min_child_weight = list(range(1, 6, 2))
        gamma = [float(i / 10.0) for i in range(0, 5)]
        n_estimators = list(range(100, 1000, 100))
        learning_rate = [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
        subsample = [i / 10.0 for i in range(6, 10)],
        col_sample_by_tree = [i / 10.0 for i in range(6, 10)]
        reg_alpha = [1e-5, 1e-2, 0.1, 1, 100, 0, 0.001, 0.005, 0.01, 0.05]

        # define grid search
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth,
                    min_child_weight=min_child_weight, gamma=gamma, colsample_bytree=col_sample_by_tree,
                    reg_alpha=reg_alpha)

        clas_reg_model_entity = ClasRegModelEntity(xgb_model, grid, Variable.typeXgb)
        self.alg_name[xgb_model] = 'xgboost.XGBRegressor'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Light GBM
    def light_gbm(self):
        # define models and parameters
        lightgbm = lgb.LGBMRegressor()
        num_leaves = list(range(50, 1000, 50))
        learning_rate = [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
        max_depth = list(range(3, 10, 2))
        max_bin = list(range(100, 1000, 100))

        # define grid search
        grid = dict(num_leaves=num_leaves, learning_rate=learning_rate, max_depth=max_depth, max_bin=max_bin)

        clas_reg_model_entity = ClasRegModelEntity(lightgbm, grid, Variable.typeLGBM)
        self.alg_name[lightgbm] = 'lgb.LGBMRegressor'

        return clas_reg_model_entity

    # -----------------------------------------------------------------------------
    # Cat Boost GBM
    @staticmethod
    def cat_boost_gbm():
        # define models and parameters
        catboost = CatBoostRegressor()
        depth = list(range(1, 11, 1))
        iterations = [250, 100, 500, 1000]
        learning_rate = [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
        l2_leaf_reg = [3, 1, 5, 10, 100]
        border_count = [32, 5, 10, 20, 50, 100, 200],
        ctr_border_count = [50, 5, 10, 20, 100, 200],
        thread_count = 4

        # define grid search
        grid = dict(depth=depth, iterations=iterations, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
                    border_count=border_count, ctr_border_count=ctr_border_count, thread_count=thread_count)

        clas_reg_model_entity = ClasRegModelEntity(catboost, grid, Variable.typeCGBM)
        # self.alg_name[catboost] = 'catboost.CatBoostRegressor'

        return clas_reg_model_entity

    '''#-----------------------------------------------------------------------------
    #HyperParameters

    epoch = 1000
    batch_size = 128
    input_dim = len(X[0])
    hidden_units_1 = 16
    hidden_units_2 = 6
    output_dim = 1
    dropout = 0.4

    def baseline_model():
        model = Sequential()
        model.add(Dense(hidden_units_1,input_dim=input_dim,activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units_2,activation='relu'))
        model.add(Dropout(dropout))
        #model.add(Dense(hidden_units_3,activation='relu'))
        #model.add(Dropout(dropout))
        model.add(Dense(output_dim,kernel_initializer='normal',activation = 'linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=1)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # compile the keras model
    # simple early stopping
    es = EarlyStopping(monitor='loss', mode='min', verbose=1)

    # fit the keras model to the dataset
    model.fit(X, Y, epochs=epoch, batch_size=batch_size)	

    # evaluate the keras model
    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy*100))'''

    def get_all_models(self):
        return [self.rid_reg(), self.lin_reg(),
                self.knn()]  # , self.des_tree(), self.svr()]#, self.ran_for(), self.adaBoost(), self.grad_boost(),
        # self.xgb(), self.light_gbm(), self.cat_boost_gbm()]

    def get_algorithm_name(self):
        return self.alg_name
