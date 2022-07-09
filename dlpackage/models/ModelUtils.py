# IMPORT
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Embedding, Input, Flatten, \
    BatchNormalization, Conv1D, MaxPooling1D, Concatenate, Input
from aipackage.dlpackage.models.attention import NLPAttention, AttentionLayer
from tensorflow.keras.initializers import Constant
from aipackage.dlpackage.PackageVariable import Variable
from tensorflow.keras.models import Model
from aipackage.dlpackage.nlppackage.Entities import InfEncoderEntity, InfDecoderEntity, InferenceEntity


class ModelUtils:

    @staticmethod
    def get_nlp_embedding_layer(embedding_entity, embedding_name=''):
        # print("Model utils embedding layer \n")
        embedding_matrix = embedding_entity.embedding_matrix
        vocab_size = embedding_entity.vocab_size
        embedding_dim = embedding_entity.embedding_dim

        initializer = 'uniform'
        trainable = type(embedding_matrix).isInstance(type(None))
        if not trainable:
            # print("\nEMBEDDING MATRIX IS PRESENT\n")
            vocab_size = embedding_matrix.shape[0]
            embedding_dim = embedding_matrix.shape[1]
            initializer = Constant(embedding_matrix)

        return Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=embedding_entity.input_length,
                         embeddings_initializer=initializer, mask_zero=True, trainable=trainable, name=embedding_name)

    @staticmethod
    def get_dense_layer(target_size, final_activation):
        return Dense(target_size, activation=final_activation)

    @staticmethod
    def get_drop_out_layer(hp, i, prefix):
        rnn_dropout = hp.Float(prefix + str(i), min_value=0.1, max_value=0.7, default=0.5, step=0.1)
        return Dropout(rate=rnn_dropout)

    def get_rnn_dropout_layer(self, hp, i):
        return self.get_drop_out_layer(hp, i, 'rnn_dropout_')

    def get_cnn_dropout_layer(self, hp, i):
        return self.get_drop_out_layer(hp, i, 'cnn_dropout_')

    def get_ann_dropout_layer(self, hp, i):
        return self.get_drop_out_layer(hp, i, 'ann_dropout_')

    def get_bi_lstm_layer(self, hp, i):
        return Bidirectional(self.get_lstm_layer(hp, i, 'bi_lstm_units_'))

    def get_bi_gru_layer(self, hp, i):
        return Bidirectional(self.get_gru_layer(hp, i, 'bi_gru_units_'))

    def get_lstm_layer(self, hp, i, prefix='lstm_units_'):
        return LSTM(units=self.get_rnn_units(hp, prefix, i), return_sequences=True)

    def get_gru_layer(self, hp, i, prefix='gru_units_'):
        return GRU(units=self.get_rnn_units(hp, prefix, i), return_sequences=True)

    @staticmethod
    def get_rnn_units(hp, prefix, i):
        return hp.Int(prefix + str(i), min_value=Variable.rnn_units_min_value, max_value=Variable.rnn_units_max_value,
                      step=Variable.rnn_units_step_value, default=Variable.rnn_units_default_value)

    @staticmethod
    def get_nlp_attention_layer():
        return NLPAttention(return_sequences=True)

    @staticmethod
    def get_batch_normalization():
        return BatchNormalization()

    @staticmethod
    def get_conv_layer(hp, i):
        cnn_padding = hp.Choice('cnn_padding_' + str(i), values=['same'])
        cnn_strides = hp.Int('cnn_strides_' + str(i), min_value=1, max_value=100, default=1, step=1)
        cnn_kernel_size = hp.Int('cnn_kernel_sizes_' + str(i), min_value=1, max_value=100, default=1, step=1)
        cnn_filters = hp.Int('cnn_filters_' + str(i), min_value=1, max_value=100, default=1, step=1)
        return Conv1D(cnn_filters, cnn_kernel_size, padding=cnn_padding, strides=cnn_strides)

    @staticmethod
    def get_max_pool_layer(hp, i):
        cnn_max_pool_strides = hp.Int('cnn_max_pool_strides_' + str(i), min_value=2, max_value=100, default=2, step=1)
        cnn_max_pool_padding = hp.Choice('cnn_max_pool_padding_' + str(i), values=['same'])
        cnn_max_pool_size = hp.Int('cnn_max_pool_size_' + str(i), min_value=1, max_value=100, default=1, step=1)
        return MaxPooling1D(pool_size=cnn_max_pool_size, padding=cnn_max_pool_padding, strides=cnn_max_pool_strides)

    @staticmethod
    def get_nlp_range():
        return Variable.nlp_layer_count

    @staticmethod
    def get_ann_range():
        return Variable.ann_layer_count

    @staticmethod
    def get_ann_input_layer(input_length):
        return Input(shape=(input_length,))

    def get_ann_dense_layer(self, hp, i, prefix='dense'):
        return Dense(units=self.get_ann_units(hp, i, prefix), activation=self.get_ann_activation_func(hp, i, prefix))

    @staticmethod
    def get_ann_units(hp, i, prefix):
        return hp.Int(prefix + '_units_' + str(i), min_value=Variable.ann_units_min_value,
                      max_value=Variable.ann_units_max_value, step=Variable.ann_units_step_value,
                      default=Variable.ann_units_default_value)

    @staticmethod
    def get_ann_activation_func(hp, i, prefix):
        return hp.Choice(prefix + '_activation_' + str(i), values=['relu', 'tanh', 'sigmoid'], default='relu')

    @staticmethod
    def get_encoder_input_layer(input_length):
        return Input(shape=(input_length,), name=Variable.encoder_input_name)

    @staticmethod
    def get_ed_attention_layer():
        return AttentionLayer(name=Variable.attention_name)

    @staticmethod
    def get_ed_dense_layer(target_size, final_activation):
        return Dense(target_size, activation=final_activation, name=Variable.dense_name)

    @staticmethod
    def get_decoder_input_layer():
        return Input(shape=(None,), name=Variable.decoder_input_name)

    def get_decoder_range(self, hp, name):
        return self.get_encoder_decoder_layer_range(hp, name)

    def get_encoder_range(self, hp, name):
        return self.get_encoder_decoder_layer_range(hp, name)

    @staticmethod
    def get_encoder_decoder_layer_range(hp, name):
        # return hp.Int(name, Variable.enc_dec_layer_start, Variable.enc_dec_layer_end)
        return Variable.enc_dec_layer_count

    def get_encoder_embedding_layer(self, embedding_entity):
        return self.get_nlp_embedding_layer(embedding_entity, Variable.encoder_embedding_name)

    @staticmethod
    def get_de_embed_layer(vocab_size, output_dim, input_length):
        return Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length, mask_zero=True,
                         name=Variable.decoder_embedding_name)

    @staticmethod
    def get_concat_layer():
        return Concatenate(axis=-1, name=Variable.concat_name)

    def get_ed_bi_lstm_layer(self, units, rnn_name):
        return Bidirectional(self.get_ed_lstm_layer(units), name=rnn_name)

    def get_ed_bi_gru_layer(self, units, rnn_name):
        return Bidirectional(self.get_ed_gru_layer(units), name=rnn_name)

    @staticmethod
    def get_ed_lstm_layer(units, rnn_name=''):
        return LSTM(units=units, return_sequences=True, return_state=True, name=rnn_name)

    @staticmethod
    def get_ed_gru_layer(units, rnn_name=''):
        return GRU(units=units, return_sequences=True, return_state=True, name=rnn_name)

    @staticmethod
    def get_ed_model(en_inp_layer, de_inp_layer, dense_output, model_entity):
        model = Model([en_inp_layer, de_inp_layer], dense_output, name=model_entity.modelName)
        model.compile(optimizer=model_entity.optimizer, loss=model_entity.loss, metrics=model_entity.metrics)
        model.summary()

        return model

    def process_and_get_ed_model(self, en_inp_layer, de_inp_layer, encoder_output, decoder_output, model_entity):
        attnOutputs = self.get_ed_attention_layer()([encoder_output, decoder_output])
        deConcatOutput = self.get_concat_layer()([decoder_output, attnOutputs[0]])
        denseOutput = self.get_ed_dense_layer(model_entity.target_size, model_entity.final_activation)(deConcatOutput)

        return self.get_ed_model(en_inp_layer, de_inp_layer, denseOutput, model_entity)

    @staticmethod
    def get_inf_encoder_model(inf_encoder_entity):
        return Model(inputs=inf_encoder_entity.encoder_input, outputs=inf_encoder_entity.encoder_rnn_output)

    @staticmethod
    def get_de_hidden_state_inp(input_length, units):
        return Input(shape=(input_length, units))

    @staticmethod
    def get_inf_decoder_outputs(inf_decoder_entity, de_initial_state):
        # Get the embeddings of the decoder sequence
        deEmbInput = inf_decoder_entity.decoder_embedding_layer(inf_decoder_entity.decoder_input)

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        return inf_decoder_entity.decoder_rnn_layer(deEmbInput, initial_state=de_initial_state)

    def process_and_get_inf_decoder_model(self, inference_entity, de_hidden_state_units, de_initial_state):
        deHiddenStateInp = self.get_de_hidden_state_inp(inference_entity.input_length, de_hidden_state_units)
        decoderOutputs = self.get_inf_decoder_outputs(inference_entity.infDecoderEntity, de_initial_state)
        # attention inference
        attnOutputs = inference_entity.attention_layer([deHiddenStateInp, decoderOutputs[0]])
        deConcatOutput = Concatenate(axis=-1)([decoderOutputs[0], attnOutputs[0]])
        # A dense softmax layer to generate prob dist. over the target vocabulary
        denseOutputs = inference_entity.dense_layer(deConcatOutput)
        # Final decoder model
        return Model([inference_entity.infDecoderEntity.decoder_input] + [deHiddenStateInp] + de_initial_state,
                     [denseOutputs] + decoderOutputs[1:])

    def get_inference_model(self, inference_entity, de_hidden_state_inp, de_initial_state):
        return self.get_inf_encoder_model(inference_entity.infEncoderEntity), self.process_and_get_inf_decoder_model(
            inference_entity, de_hidden_state_inp, de_initial_state)

    @staticmethod
    def get_inference_entity(model):
        encoder_input = model.get_layer(Variable.encoder_input_name).input
        decoder_input = model.get_layer(Variable.decoder_input_name).input
        decoder_embedding_layer = model.get_layer(Variable.decoder_embedding_name)
        attention_layer = model.get_layer(Variable.attention_name)
        dense_layer = model.get_layer(Variable.dense_name)
        input_length = decoder_embedding_layer.input_length

        rnnLayerName = list(filter(lambda x: x[-1].isnumeric(), list(map(lambda x: x.name, model.layers))))
        print(rnnLayerName)

        encoder_rnn_output = model.get_layer(list(filter(lambda x: x.startswith("encoder"), rnnLayerName))[-1]).output
        decoder_rnn_layer = model.get_layer(rnnLayerName[-1])
        if Variable.bidirectionalName in decoder_rnn_layer.__class__.__name__:
            units = model.get_layer(rnnLayerName[-1]).layer.units
        else:
            units = model.get_layer(rnnLayerName[-1]).units

        infEncoderEntity = InfEncoderEntity(encoder_input, encoder_rnn_output)
        infDecoderEntity = InfDecoderEntity(decoder_input, decoder_embedding_layer, decoder_rnn_layer)

        return InferenceEntity(infEncoderEntity, infDecoderEntity, attention_layer, dense_layer, input_length, units)
