class Variable:
    allPickleFolderName = 'data/allPickleModel'
    bestPickleFolderName = 'data/bestPickleModel'
    locationSeparator = '/'
    geneFolderName = 'data/geneData'
    dataFolderName = 'data'
    writeBinary = 'wb'
    readBinary = 'rb'
    modelFileName = 'data/models.feather'
    edaFileName = 'data/EDA/eda.feather'
    datasetLocation = '../../../Dataset/'
    tempDatasetLocation = '../Dataset/'
    submissionFileName = 'submission.csv'
    fileSeparator = '_'
    featherFolderName = 'Feather'
    featherTrainFileName = '/Train.feather'
    featherTestFileName = '/Test.feather'
    csvTrainFileName = 'Train.csv'
    csvTestFileName = 'Test.csv'
    allVisualizerFolderName = 'data/allVisualizer'
    learningCurve = 'learning_curve'
    validationCurve = 'validation_curve'
    edaLocation = 'data/EDA'

    EXTRAS_TRAIN = 'TRAIN'
    EXTRAS_TEST = 'TEST'

    # EXTENSION VARIABLE

    pythonExtension = '.py'
    pickleExtension = '.pkl'
    htmlExtension = '.html'

    pandasProfiling = '_pandas_profiling'
    sweetviz = '_sweetviz'
    dataPrep = '_dataprep'
    autoviz = '_autoviz'
    dtale = '_dtale'

    # HYPER PARAMETER VARIABLE

    typeGrid = 'grid'
    typeBayes = 'bayes'
    typeGene = 'gene'
    typeOptuna = 'optuna'
    typeGA = 'sklearnGenetic'

    # ML TYPES

    typeClassification = 'Classification'
    typeRegress = 'Regression'
    typeMultiClass = 'MultiClassification'
    typeSegmentation = 'Segmentation'

    # MODEL VARIABLE

    typeLinReg = 'LINEAR_REGRESSION_'
    typeLogReg = 'LOGISTIC_REGRESSION_'
    typeRidge = 'RIDGE_'
    typeKnn = 'KNN_'
    typeSvc = 'SVC_'
    typeDesTree = 'DECISION_TREE_'
    typeRanFor = 'RANDOM_FOREST_'
    typeGradBoost = 'GRADIENT_BOOSTING_'
    typeAdaBoost = 'ADAPTIVE_BOOSTING_'
    typeXgb = 'XGB_'
    typeLGBM = 'LIGHT_GBM_'
    typeCGBM = 'CAT_BOOST_GBM_'
    typeGaussianNB = 'GAUSSIAN_NB_'
    typeBerNB = 'BERNOULLI_NB_'
    typeKmeans = 'KMEANS_'
    typeAglo = 'AGGLOMERATIVE_'
    typeBirch = 'BIRCH_'
    typeMiniBatch = 'MINI_BATCH_'
    typeSpecCluster = 'SPEC_CLUSTER_'
    typeGaussianMix = 'GAUSSIAN_MIX_'

    scoreName = 'Best_Score'
    valScoreName = 'Best_Val_Score'

    isRunAllFileEnabled = True
    runAllFileLimitStart = 0
    runAllFileLimitEnd = 1000
