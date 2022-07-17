# IMPORT
import matplotlib
import numpy as np
import optuna
import wandb
from aiautomation.mlpackage.Entities import AccuracyEntity
from aiautomation.mlpackage.PackageVariable import Variable
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.extmath import softmax
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous
from tpot import TPOTClassifier, TPOTRegressor
from yellowbrick.model_selection import LearningCurve, ValidationCurve


class HyperParameterTuning:

    def __init__(self, hp_entity):
        self.hPEntity = hp_entity
        print("X_TRAIN SHAPE = ", hp_entity.x_train.shape)
        print("Y_TRAIN SHAPE = ", hp_entity.y_train.shape)
        self.best_score = 0
        self.best_val_score = 0
        self.io = None
        self.actualFolderName = ""
        # wandb.init(project="visualize-sklearn")

    def set_actual_folder_name(self, folder_name):
        self.actualFolderName = folder_name

    def init_wand_b(self, name, custom_dir):
        wandb.init(project=self.actualFolderName, name=name, save_code=True, dir=custom_dir)

    def update_io(self, io):
        self.io = io

    # search space for bayesian
    @staticmethod
    def search_space(grid):
        space = {}

        for i in grid.keys():
            param = hp.choice(str(i), grid[i])
            space[i] = param
        return space

        # search space for bayesian

    @staticmethod
    def ga_search_param(grid):
        space = {}

        for i in grid.keys():
            if isinstance(grid[i], list) and isinstance(grid[i][0], float):
                param = Continuous(grid[i][0], grid[i][-1])
            else:
                param = Categorical(grid[i])
            space[i] = param
        return space

    def set_scores(self, score, val_score):
        self.best_score = score
        self.best_val_score = val_score

    @staticmethod
    def get_bayes_param(best, grid):
        new_grid = {}
        for i in best:
            new_grid[i] = grid[i][best[i]]
        return new_grid

    def set_scoring(self, scoring):
        self.hPEntity.scoring = scoring

    def get_all_scoring(self):
        return self.hPEntity.scoring

    ###################################################################################################################
    # Fit the model
    # GRID SEARCH
    def model_grid_fit(self, model_entity, file_name, folder_path):
        grid_search = GridSearchCV(estimator=model_entity.alg, param_grid=model_entity.grid, n_jobs=1,
                                   cv=self.hPEntity.cv, scoring=self.hPEntity.scoring, error_score=0)

        grid_search.fit(self.hPEntity.x_train, self.hPEntity.y_train)

        score = 0
        val_score = 0
        if self.hPEntity.model_type != Variable.typeSegmentation:
            score = grid_search.best_score_
            val_score = grid_search.score(self.hPEntity.x_val, self.hPEntity.y_val)

        self.run_all_visualization(grid_search, self.hPEntity.cv, self.hPEntity.scoring, self.hPEntity.x_train,
                                   self.hPEntity.y_train, self.hPEntity.x_val, self.hPEntity.y_val, model_entity.grid,
                                   file_name,
                                   self.hPEntity.model_type, folder_path)

        self.check_and_save_model(score, val_score, file_name, folder_path, grid_search)

        return AccuracyEntity(file_name, score, val_score, str(grid_search.best_params_))

    # BAYESIAN OPTIMIZATION # https://www.kaggle.com/code/prashant111/bayesian-optimization-using-hyperopt/notebook
    def model_bayes_fit(self, model_entity, file_name, folder_path):

        x_train = self.hPEntity.x_train
        y_train = self.hPEntity.y_train
        model = model_entity.alg
        grid = model_entity.grid

        # Objective
        def objective(params):
            temp_score = cross_val_score(model, x_train, y_train, cv=self.hPEntity.cv,
                                         scoring=self.hPEntity.scoring).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -temp_score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective,
                    algo=tpe.suggest,
                    max_evals=80,
                    space=self.search_space(grid),
                    trials=trials)

        # setting the params for model
        param = self.get_bayes_param(best, grid)
        alg = model.set_params(**param)
        alg.fit(x_train, y_train)

        score = 0
        val_score = 0
        if self.hPEntity.model_type != Variable.typeSegmentation:
            score = trials.best_trial['result']['loss']
            val_score = alg.score(self.hPEntity.x_val, self.hPEntity.y_val)

        self.run_all_visualization(alg, self.hPEntity.cv, self.hPEntity.scoring, self.hPEntity.x_train,
                                   self.hPEntity.y_train, self.hPEntity.x_val, self.hPEntity.y_val, grid, file_name,
                                   self.hPEntity.model_type, folder_path)

        self.check_and_save_model(score, val_score, file_name, folder_path, alg)

        return AccuracyEntity(file_name, score, val_score, str(best))

    # GENETIC ALGORITHM
    def model_gene_fit(self, model_entity, file_name, folder_path):

        """accuracy, average_precision, roc_auc, recall"""
        x_train = self.hPEntity.x_train
        y_train = self.hPEntity.y_train
        cv = self.hPEntity.cv
        scoring = self.hPEntity.scoring
        model_type = self.hPEntity.model_type

        if model_type == Variable.typeMultiClass:
            model = model_entity.gene_alg
            grid = model_entity.gene_grid
        else:
            model = model_entity.alg
            grid = model_entity.grid

        if model_type == Variable.typeRegress:
            tpot = TPOTRegressor(generations=5, population_size=24, offspring_size=12, verbosity=2, early_stop=12,
                                 config_dict={self.hPEntity.alg_name[model]: grid}, cv=cv, scoring=scoring)
        else:
            tpot = TPOTClassifier(generations=5, population_size=24, offspring_size=12, verbosity=2, early_stop=12,
                                  config_dict={self.hPEntity.alg_name[model]: grid}, cv=cv, scoring=scoring)

        model = tpot.fit(x_train, y_train)

        # TODO REMOVE THE BELOW COMMENTED CODE AND TRY TO GET GENE FILE PATH LOCATION ACCORDING TO RUN ALL FILE
        self.export_gene(model, file_name)

        score = 0
        val_score = 0
        if model_type != Variable.typeSegmentation:
            score = model.score(x_train, y_train)
            val_score = model.score(self.hPEntity.x_val, self.hPEntity.y_val)

        self.run_all_visualization(model, self.hPEntity.cv, self.hPEntity.scoring, self.hPEntity.x_train,
                                   self.hPEntity.y_train, self.hPEntity.x_val, self.hPEntity.y_val, grid, file_name,
                                   self.hPEntity.model_type, folder_path)

        self.check_and_save_model(score, val_score, file_name, folder_path, model)

        return AccuracyEntity(file_name, score, val_score, "example")

    # OPTUNA ALGORITHM
    def model_optuna_fit(self, model_entity, file_name, folder_path):

        model = model_entity.alg
        grid = model_entity.grid
        x_train = self.hPEntity.x_train
        y_train = self.hPEntity.y_train

        def objective(temp_trial):
            space = {}
            for i in grid.keys():
                space[i] = temp_trial.suggest_categorical(str(i), grid[i])
            model.set_params(**space)
            temp_score = cross_val_score(model, x_train, y_train, cv=self.hPEntity.cv,
                                         scoring=self.hPEntity.scoring).mean()
            return temp_score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        trial = study.best_trial

        # setting the params for model
        alg = model.set_params(**trial.params)
        alg.fit(x_train, y_train)

        score = 0
        val_score = 0
        if self.hPEntity.model_type != Variable.typeSegmentation:
            score = trial.value
            val_score = alg.score(self.hPEntity.x_val, self.hPEntity.y_val)

        self.run_all_visualization(alg, self.hPEntity.cv, self.hPEntity.scoring, self.hPEntity.x_train,
                                   self.hPEntity.y_train, self.hPEntity.x_val, self.hPEntity.y_val, grid, file_name,
                                   self.hPEntity.model_type, folder_path)

        self.check_and_save_model(score, val_score, file_name, folder_path, alg)

        return AccuracyEntity(file_name, score, val_score, str(trial.params))

    # GENETIC ALGORITHM
    def model_sklearn_genetic_fit(self, model_entity, file_name, folder_path):
        ga_search = GASearchCV(estimator=model_entity.alg, param_grid=self.ga_search_param(model_entity.grid), n_jobs=1,
                               cv=self.hPEntity.cv, scoring=self.hPEntity.scoring)

        ga_search.fit(self.hPEntity.x_train, self.hPEntity.y_train)
        model = ga_search.best_estimator_

        score = 0
        val_score = 0
        if self.hPEntity.model_type != Variable.typeSegmentation:
            score = model.score(self.hPEntity.x_train, self.hPEntity.y_train)
            val_score = model.score(self.hPEntity.x_val, self.hPEntity.y_val)

        self.run_all_visualization(model, self.hPEntity.cv, self.hPEntity.scoring, self.hPEntity.x_train,
                                   self.hPEntity.y_train, self.hPEntity.x_val, self.hPEntity.y_val, model_entity.grid,
                                   file_name,
                                   self.hPEntity.model_type, folder_path)

        self.check_and_save_model(score, val_score, file_name, folder_path, model)

        return AccuracyEntity(file_name, score, val_score, str(ga_search.best_params_))

    def check_and_save_model(self, score, val_score, model_name, folder_path, model):
        self.save_all_model(model_name, folder_path, model)
        if self.can_save_model(score, val_score):
            self.save_best_model(model_name, model)
            self.best_score = score
            self.best_val_score = val_score

    def export_gene(self, tpot, file_name):
        self.io.export_gene(tpot, file_name)

    def save_best_model(self, model_name, model):
        self.io.save_best_model(model_name, model)

    def save_all_model(self, model_name, folder_path, model):
        self.io.save_all_model(model_name, folder_path, model)

    def save_all_visualizer(self, model_name, folder_path, model):
        self.io.save_all_visualizer(model_name, folder_path, model)

    def can_save_model(self, score, val_score):
        if score > 0 and val_score > 0:
            if self.get_check_score(self.best_val_score, val_score):
                return True
            if self.best_val_score == val_score:
                if self.get_check_score(self.best_score, score) or self.best_score == score:
                    return True

    def get_check_score(self, best_score, score):
        if self.hPEntity.model_type == Variable.typeRegress:
            return best_score > score
        else:
            return best_score < score

    def start_visualization(self, model, cv, scoring, x_train, y_train, x_val, y_val, grid, model_name, folder_path):

        matplotlib.use('Agg')
        self.start_learning_curve(model, cv, scoring, x_train, y_train, model_name, folder_path)
        print("\n")
        # self.startValidationCurve(model, cv, scoring, x_val, y_val, grid, modelName, folderPath)

    def start_learning_curve(self, model, cv, scoring, x, y, model_name, folder_path):
        visualizer = LearningCurve(model, cv=cv, scoring=scoring, n_jobs=4)
        visualizer.fit(x, y)  # Fit the data to the visualizer
        model_name = model_name + Variable.fileSeparator + Variable.learningCurve
        self.save_all_visualizer(model_name, folder_path, visualizer)
        # visualizer.savefig("mygraph.png")

    def start_validation_curve(self, model, cv, scoring, x, y, grid, model_name, folder_path):
        for key, value in grid.items():
            visualizer = ValidationCurve(model, cv=cv, scoring=scoring, n_jobs=4, param_name=key, param_range=value)
            visualizer.fit(x, y)  # Fit the data to the visualizer
            model_name = model_name + Variable.fileSeparator + key + Variable.fileSeparator + Variable.validationCurve
            self.save_all_visualizer(model_name, folder_path, visualizer)
        # visualizer.show()

    def start_wandb(self, model, x_train, y_train, x_val, y_val, model_name, model_type):
        wandb_dir = self.io.storeDataDirName + Variable.locationSeparator + Variable.dataFolderName
        self.init_wand_b(model_name, wandb_dir)
        wandb.sklearn.plot_learning_curve(model, x_train, y_train)

        if model_type == Variable.typeSegmentation:
            # All clustering plots
            wandb.sklearn.plot_clusterer(model, x_train, cluster_labels, labels=None, model_name=model_name)

        elif model_type == Variable.typeRegress:
            # All regression plots
            wandb.sklearn.plot_regressor(model, x_train, x_val, y_train, y_val, model_name=model_name)

        else:
            # Visualize all classifier plots
            wandb.sklearn.plot_classifier(model, x_train, x_val, y_train, y_val, y_pred=model.predict(x_val),
                                          y_probas=self.get_predict_proba(model, x_val), labels=self.hPEntity.labels,
                                          model_name=model_name, feature_names=None)

        wandb.finish()

    @staticmethod
    def get_predict_proba(model, x_val):
        try:
            return model.predict_proba(x_val)
        except:
            d = model.decision_function(x_val)
            d_2d = np.c_[-d, d]
            return softmax(d_2d)

    def run_all_visualization(self, model, cv, scoring, x_train, y_train, x_val, y_val, grid, model_name, model_type,
                              folder_path):
        print("\nstart Visualization\n")
        self.start_visualization(model, cv, scoring, x_train, y_train, x_val, y_val, grid, model_name, folder_path)
        self.start_wandb(model, x_train, y_train, x_val, y_val, model_name, model_type)
        print("\nfinished Visualization\n")
