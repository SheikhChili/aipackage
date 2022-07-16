# IMPORT
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from aiautomation.mlpackage.Entities import ClasRegModelEnitity
from aiautomation.mlpackage.PackageVariable import Variable


class Models:

    def __init__(self, cluster):
        self.alg_name = {}
        self.clusters = cluster

    # define different models
    def agglo_cluster(self):
        # define models and parameters
        model = AgglomerativeClustering(n_clusters=self.clusters)

        # Whole values
        affinity = ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"]
        compute = ["auto", True, False]
        linkage = ["ward", "complete", "average", "single"]
        threshold = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9]
        distance = [True, False]

        grid = dict(affinity=affinity, compute_full_tree=compute, linkage=linkage, compute_distances=distance,
                    distance_threshold=threshold)

        clas_reg_model_entity = ClasRegModelEnitity(model, grid, Variable.typeAglo)
        self.alg_name[model] = 'sklearn.cluster.AgglomerativeClustering'

        return clas_reg_model_entity

    # Balanced Iterative Reducing and Clustering
    def birch(self):
        # define models and parameters
        model = Birch(n_clusters=self.clusters)

        threshold = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9]
        branching_factor = list(range(5, 100, 5))
        labels = [True, False]

        grid = dict(threshold=threshold, branching_factor=branching_factor, compute_labels=labels)

        clas_reg_model_entity = ClasRegModelEnitity(model, grid, Variable.typeBirch)
        self.alg_name[model] = 'sklearn.cluster.Birch'

        return clas_reg_model_entity

    # Kmeans
    def kmeans(self):
        # define the model
        model = KMeans(n_clusters=self.clusters, n_jobs=-1)

        n_init = list(range(0, 100, 5))
        # max_iter = list(range(0, 1000, 50))
        tol = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        distance = ["auto", True, False]
        algorithm = ["auto", "full", "elkan"]

        grid = dict(n_init=n_init, tol=tol, algorithm=algorithm, precompute_distances=distance)

        clas_reg_model_entity = ClasRegModelEnitity(model, grid, Variable.typeKmeans)
        self.alg_name[model] = 'sklearn.cluster.KMeans'

        return clas_reg_model_entity

    # Mini-Batch K-Means
    def min_batch_k(self):
        # define the model
        model = MiniBatchKMeans(n_clusters=self.clusters)

        max_iter = list(range(0, 1000, 50))
        batch_size = list(range(0, 1000, 50))
        labels = [True, False]
        tol = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        improvement = list(range(0, 100, 5))
        reassignment = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                        0.8, 0.9]

        grid = dict(max_iter=max_iter, batch_size=batch_size, tol=tol, compute_labels=labels,
                    max_no_improvement=improvement, reassignment_ratio=reassignment)

        clas_reg_model_entity = ClasRegModelEnitity(model, grid, Variable.typeMiniBatch)
        self.alg_name[model] = 'sklearn.cluster.MiniBatchKMeans'

        return clas_reg_model_entity

    # Spectral Clustering
    def spec_cluster(self):
        # define the model
        model = SpectralClustering(n_clusters=self.clusters, n_jobs=-1)

        eigen_solver = ["arpack", "lobpcg", "amg", None]
        n_init = list(range(0, 100, 5))
        gamma = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                 0.9, 1.0]
        n_neighbors = list(range(0, 100, 5))
        eigen_tol = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9, 1.0]
        labels = ["kmeans", "discretize"]
        affinity = ["nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"]

        grid = dict(eigen_solver=eigen_solver, n_init=n_init, gamma=gamma, affinity=affinity, n_neighbors=n_neighbors,
                    eigen_tol=eigen_tol, assign_labels=labels)

        clas_reg_model_entity = ClasRegModelEnitity(model, grid, Variable.typeSpecCluster)
        self.alg_name[model] = 'sklearn.cluster.SpectralClustering'

        return clas_reg_model_entity

    # Gaussian Mixture Model
    def gaussian_mix(self):
        # define the model
        model = GaussianMixture(n_components=self.clusters)

        covariance_type = ["full", "tied", "diag", "spherical"]
        tol = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        reg_covar = [0.0, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9]
        # max_iter = list(range(0, 1000, 50))
        n_init = list(range(0, 100, 5))
        init_params = ["kmeans", "random"]
        warm_start = [True, False]

        grid = dict(covariance_type=covariance_type, tol=tol, reg_covar=reg_covar, n_init=n_init,
                    init_params=init_params, warm_start=warm_start)

        clas_reg_model_entity = ClasRegModelEnitity(model, grid, Variable.typeGaussianMix)
        self.alg_name[model] = 'sklearn.mixture.GaussianMixture'

        return clas_reg_model_entity

    def get_all_models(self):
        return [self.birch(), self.min_batch_k(), self.gaussian_mix(), self.spec_cluster(), self.kmeans(),
                self.agglo_cluster()]

    def get_algorithm_name(self):
        # alg_name = self.alg_name
        # del self.alg_name
        return self.alg_name
