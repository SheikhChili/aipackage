class HyperParamEntity:

    def __init__(self, x_train=[], y_train=[], x_val=[], y_val=[], alg_name={}, cv=None, model_type="", scoring="",
                 labels=[]):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.scoring = scoring
        self.cv = cv
        self.model_type = model_type
        self.alg_name = alg_name
        self.labels = labels


class ClasRegModelEntity:
    def __init__(self, alg, grid, model_name):
        self.alg = alg
        self.grid = grid
        self.model_name = model_name


class MultiClasModelEntity:
    def __init__(self, alg, grid, model_name, gene_grid, gene_alg):
        self.alg = alg
        self.grid = grid
        self.model_name = model_name
        self.gene_grid = gene_grid
        self.gene_alg = gene_alg


class AccuracyEntity:

    def __init__(self, file_name, score=0, val_score=0, param="", metrics="", error_msg=""):
        self.filename = file_name
        self.score = score
        self.val_score = val_score
        self.param = param
        self.metrics = metrics
        self.error_msg = error_msg


class SubmissionEntity:
    def __init__(self, predictions=[], id_=[], id_2=[], id_3=[], fields=[], file_name=None):
        self.predictions = predictions
        self.id_ = id_
        self.id_2 = id_2
        self.id_3 = id_3
        self.fields = fields
        self.fileName = file_name
