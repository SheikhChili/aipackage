import tensorflow as tf


class Variable:
    locationSeparator = "/"
    filenameSeparator = "_"
    dataFolderName = "data"
    preprocessFolderName = "data/preprocessData"
    searchDataFolderName = "data/searchData"
    ranSearchFolderName = "data/searchData/ranSearchData"
    baySearchFolderName = "data/searchData/baySearchData"
    hpSearchFolderName = "data/searchData/hyperBandSearchData"
    bestPredictModelFolderName = "data/bestPredictModelData"
    allPredictModelFolderName = "data/allPredictModelData"
    trainPreprocessFileName = "trainCleanedText.feather"
    testPreprocessFileName = "testCleanedText.feather"
    dataUniqueWordFileName = "dataUniqueWords.feather"
    dataWordToIntFileName = "dataWordToInt.txt"
    labelFileName = "labelText.feather"
    labelUniqueWordFileName = "labelUniqueWords.feather"
    labelWordToIntFileName = "labelWordToInt.txt"
    writeBinary = "wb"
    readBinary = "rb"
    modelFileName = "data/models.feather"
    predictFileName = "data/predict.feather"
    nlpDatasetLocation = "../../../../Dataset/NLP/"
    audioDatasetLocation = "../../../Dataset/Audio/"
    submissionFileName = "submission.csv"

    featherFolderName = "Feather"
    featherTrainFileName = "/Train.feather"
    featherTestFileName = "/Test.feather"
    csvTrainFileName = "Train.csv"
    csvTestFileName = "Test.csv"

    trainFileName = locationSeparator + featherFolderName + featherTrainFileName
    testFileName = locationSeparator + featherFolderName + featherTestFileName

    wordDataLocation = "WordData/"
    stopWordFileLocation = nlpDatasetLocation + wordDataLocation + "Stopwords.txt"
    contractionFileLocation = nlpDatasetLocation + wordDataLocation + "ContractionMapping.txt"
    fileNamePrefix = preprocessFolderName + locationSeparator
    embeddingFileNamePrefix = "glove.6B."
    embeddingFileNameSuffix = "d.txt"
    embeddingFileLocationPrefix = nlpDatasetLocation + wordDataLocation

    trainDataLimit = 100
    testDataLimit = 1

    # EXTENSION VARIABLE
    modelExtension = ".h5"
    pickleExtension = ".pkl"

    typeRandom = "random"
    typeBayes = "bayes"
    typeHyperBand = "hyper_band"

    typeEmbedding = "_Embedding_"

    scoreName = "Best Accuracy"
    valScoreName = "Val Accuracy"

    # ML TYPES
    typeClassification = "Classification"
    typeRegression = "Regression"
    typeEncoderDecoder = "EncoderDecoder"
    typeAudio = "Audio"
    typeText = "TEXT"
    typeImage = "IMAGE"

    # DATA TYPES VARIABLE
    TYPE_EXTRAS_TRAIN = "TRAIN"
    TYPE_EXTRAS_LABEL = "LABEL"
    TYPE_EXTRAS_TEST = "TEST"
    TYPE_EXTRAS_NEW_TEST = "NEW_TEST"

    # TEXT TYPES
    TYPE_CHAR = 'char'
    TYPE_WORD = 'word'

    # TEXT GENERATION FILE NAME
    uniqueWordFileName = "UniqueWords.txt"
    wordToIntFileName = "WordToInt.txt"
    uniqueCharFileName = "uniqueChar.txt"
    charToIntFileName = "charToInt.txt"
    charDataLocation = "../../../../../../Dataset/NLP/Data/"
    trainCharFileName = "trainCharCleanedText.txt"
    labelCharFileName = "labelCharText.txt"
    scriptLocation = "Script/"
    charDataLocation = "Data/"
    textGenerateDatasetLocation = "../../../../../Dataset/NLP/"
    textGenerationTestWordsCount = 100

    objective = 'val_loss'
    max_trials = 1
    executions_per_trial = 1
    epochs = 50
    batch_size = 20
    # distribution_strategy = tf.distribute.MirroredStrategy(cross_device_ops =
    # tf.distribute.HierarchicalCopyAllReduce())
    distribution_strategy = mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
                                                                               cross_device_ops=tf.distribute
                                                                               .HierarchicalCopyAllReduce())

    # LAYER_NAME
    encoder_input_name = "encoder_input_layer"
    decoder_input_name = "decoder_input_layer"
    encoder_embedding_name = "encoder_embedding_layer"
    decoder_embedding_name = "decoder_embedding_layer"
    attention_name = "attention_layer"
    concat_name = "concat_layer"
    time_distributed_name = "time_distributed_layer"
    dense_name = "dense_layer"
    encoder_rnn_prefix = "encoder_rnn_layer_"
    decoder_rnn_prefix = "decoder_rnn_layer_"
    bidirectionalName = "Bidirectional"
    lstm_name = "LSTM"
    gru_name = "GRU"
    bidirectionalGru_name = "BidirectionalGRU"
    decoder_name = "Decoder"
    enc_dec_name = "EncoderDecoder"

    # UNITS
    embedding_output_dim_max_value = 512
    embedding_output_dim_min_value = 32
    embedding_output_dim_step_value = 32
    embedding_output_dim_default_value = 128

    enc_dec_layer_start = 3
    enc_dec_layer_end = 4

    layer_start = 1
    layer_end = 6

    nlp_layer_count = 1
    ann_layer_count = 1
    enc_dec_layer_count = 0

    rnn_units_max_value = 512
    rnn_units_min_value = 32
    rnn_units_step_value = 32
    rnn_units_default_value = 128

    ann_units_max_value = 1024
    ann_units_min_value = 32
    ann_units_step_value = 32
    ann_units_default_value = 128

    embedding_dim_array = [100]  # , 50, 200, 300]

    triangular_mode = "triangular"
    triangular2_mode = "triangular2"
    exp_mode = "exp_range"
    max_lr = 3e-4
    base_lr = 3e-6
    gamma = 0.99994
    step_size = 8

    sample_rate = 8000
