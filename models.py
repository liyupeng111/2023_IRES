# tensorflow + keras functions
# tensorflow>=2.10.0
import tensorflow as tf
from tensorflow import keras, convert_to_tensor, string
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow import linalg, ones, maximum, newaxis
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Dense, Layer, Embedding, MaxPooling1D, LSTM
from tensorflow.keras.layers import LayerNormalization, ReLU, Dropout
from tensorflow.keras.layers import Activation, Flatten, Conv1D, BatchNormalization
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

from keras import backend as K
from keras.backend import softmax

# transformers functions
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertConfig, TFBertModel

# graph neural network functions
from spektral.layers import GCNConv, ChebConv, GlobalSumPool, GlobalAvgPool, GATConv, GeneralConv, EdgeConv, ARMAConv
from spektral.utils.convolution import gcn_filter, chebyshev_filter

# self-defined functions
from functions import one_hot_encode_padding, encode_padding
from functions import LRScheduler, TransformerModel

def model_onehot_conv(seq_length=600, dropout_rate=0.1):
    
    # 1
    # one-hot encoder + cnn
    
    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=(seq_length, 4), filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model


def model_embed(seq_length=600, dropout_rate=0.1, d_model = 16, enc_vocab_size = 5):
    
    # 2
    # embedding only
     
    inputs = tf.keras.layers.Input(shape=(seq_length,))
    outputs = Embedding(input_dim=enc_vocab_size, output_dim=d_model)(inputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model_embed_conv(seq_length=600, dropout_rate=0.1, d_model = 16, enc_vocab_size = 5):
    
    # 3
    # embedding + cnn
     
    inputs = tf.keras.layers.Input(shape=(seq_length,))
    outputs = Embedding(input_dim=enc_vocab_size, output_dim=d_model)(inputs)
    outputs = Conv1D(activation="relu", input_shape=(seq_length, d_model), filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model_embed_bert(seq_length=600, dropout_rate=0.1):
    
    # 4
    # Bert (embedding + attention)
    
    d_model = 16
    
    config = BertConfig(vocab_size=10, 
                   hidden_size=d_model,
                   num_hidden_layers=4,
                   num_attention_heads=8,
                   intermediate_size =32,
                   max_position_embeddings =seq_length,
                   position_embedding_type = 'relative_key_query')

    inputs = tf.keras.layers.Input(shape=(seq_length,) ,dtype='int32')
    outputs = TFBertModel(config)(inputs)
    outputs = Flatten()(outputs.last_hidden_state)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def model_embed_bert_conv(seq_length=600, dropout_rate=0.1, d_model = 16):
    
    # 5
    # Bert + cnn
    
    config = BertConfig(vocab_size=10, 
                   hidden_size=d_model,
                   num_hidden_layers=4,
                   num_attention_heads=8,
                   intermediate_size =32,
                   max_position_embeddings =seq_length,
                   position_embedding_type = 'relative_key_query')

    inputs = tf.keras.layers.Input(shape=(seq_length,) ,dtype='int32')
    outputs = TFBertModel(config)(inputs)
    outputs = Conv1D(activation="relu", input_shape=(seq_length, d_model), filters=128, kernel_size=8, padding="same")(outputs.last_hidden_state)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model



def model_onehot_graph_attention(seq_length=600, dropout_rate=0.1):
    
    # 6
    # one-hot encoding, Graph Attention Networks
    
    n_node_features = 14
    n_edge_features = 3
    dropout = 0.1
    channels = 128
    l2_reg = 5e-4
    num_classes=1

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, 4))
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)

    graph_conv_1 = GATConv(16,  attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg),
                           use_bias=False)([X_in, fltr_in])
    graph_conv_2 = GATConv(16, attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           use_bias=False)([graph_conv_1, fltr_in])
    graph_conv_3 = GATConv(16, attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           use_bias=False)([graph_conv_2, fltr_in])
    graph_conv_4 = GATConv(16, attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           use_bias=False)([graph_conv_3, fltr_in])

    outputs = GlobalAvgPool()(graph_conv_4)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model

def model_embed_graph_attention(seq_length=600, dropout_rate=0.1, d_model = 16, enc_vocab_size = 5):
    
    # 7
    # embedding, Graph Attention Networks
    
    n_node_features = 14
    n_edge_features = 3
    l2_reg = 5e-4
    num_classes=1

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, ))
    
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)
    
    X_out = Embedding(input_dim=enc_vocab_size, output_dim=d_model, input_length=seq_length)(X_in)
    
    graph_conv_1 = GATConv(16,  attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           use_bias=False)([X_out, fltr_in])
    graph_conv_2 = GATConv(16, attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           use_bias=False)([graph_conv_1, fltr_in])
    graph_conv_3 = GATConv(16, attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           use_bias=False)([graph_conv_2, fltr_in])
    graph_conv_4 = GATConv(16, attn_heads=8,
                           dropout_rate=dropout_rate,
                           activation='relu',
                           use_bias=False)([graph_conv_3, fltr_in])

    outputs = GlobalAvgPool()(graph_conv_4)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model

def model_onehot_graph_conv(seq_length=600, dropout_rate=0.1):
    
    # 8 
    # one-hot encoding, Graph convolution Networks
    
    n_node_features = 14
    n_edge_features = 3
    dropout = 0.1
    l2_reg = 5e-4
    num_classes=1

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, 4))
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)

    graph_conv_1 = ARMAConv(128, dropout_rate=dropout_rate)([X_in, fltr_in])
    graph_conv_2 = ARMAConv(128, dropout_rate=dropout_rate)([graph_conv_1, fltr_in])
    graph_conv_3 = ARMAConv(128, dropout_rate=dropout_rate)([graph_conv_2, fltr_in])
    graph_conv_4 = ARMAConv(128, dropout_rate=dropout_rate)([graph_conv_3, fltr_in])

    outputs = GlobalAvgPool()(graph_conv_4)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model


def model_embed_graph_conv(seq_length=600, dropout_rate=0.1, d_model = 16, enc_vocab_size = 5):
    
    # 9
    # embedding, Graph convolution Networks
    
    n_node_features = 14
    n_edge_features = 3
    dropout = 0.1
    l2_reg = 5e-4
    num_classes=1
    

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, ))
    
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)
    
    X_out = Embedding(input_dim=enc_vocab_size, output_dim=d_model, input_length=seq_length)(X_in)

    graph_conv_1 = ARMAConv(128, dropout_rate=dropout_rate)([X_out, fltr_in])
    graph_conv_2 = ARMAConv(128, dropout_rate=dropout_rate)([graph_conv_1, fltr_in])
    graph_conv_3 = ARMAConv(128, dropout_rate=dropout_rate)([graph_conv_2, fltr_in])
    graph_conv_4 = ARMAConv(128, dropout_rate=dropout_rate)([graph_conv_3, fltr_in])

    outputs = GlobalAvgPool()(graph_conv_4)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model



def model_onehot_structure_conv(seq_length=600, dropout_rate=0.1):
    
    # 10
    # onehot encoder for both sequence and structure, cnn
    
    input1 = tf.keras.layers.Input(shape=(seq_length, 4)) # sequence
    input2 = tf.keras.layers.Input(shape=(seq_length, 3)) # structure

    outputs = tf.keras.layers.Concatenate()([input1, input2])

    outputs=Conv1D(activation="relu", input_shape=(seq_length, 7), filters=128, kernel_size=8, padding="same")(outputs)
    outputs=MaxPooling1D()(outputs)
    outputs=Dropout(dropout_rate)(outputs)
    outputs=Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs=MaxPooling1D()(outputs)
    outputs=Dropout(dropout_rate)(outputs)
    outputs=Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs=MaxPooling1D()(outputs)
    outputs=Dropout(dropout_rate)(outputs)
    outputs=Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs=MaxPooling1D()(outputs)
    outputs=Dropout(dropout_rate)(outputs)
    outputs=Flatten()(outputs)
    outputs=Dense(32)(outputs)
    outputs=Activation('relu')(outputs)
    outputs=Dropout(dropout_rate)(outputs)
    outputs=Dense(1)(outputs)
    outputs=Activation('sigmoid')(outputs)
    
    model = keras.Model(inputs=[input1, input2], outputs=outputs)
    
    return model





    
def model1(seq_length=600, dropout_rate=0.1):
    
    # transformer
    enc_vocab_size = 5 # Vocabulary size for the encoder
    dec_vocab_size = enc_vocab_size # Vocabulary size for the decoder

    enc_seq_length = seq_length  # Maximum length of the input sequence
    dec_seq_length = seq_length  # Maximum length of the target sequence

    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 32  # Dimensionality of the inner fully connected layer
    d_model = 16  # Dimensionality of the model sub-layers' outputs
    n = 1  # Number of layers in the encoder stack
    
    word_embedding_layer = Embedding(input_dim=enc_vocab_size, output_dim=d_model)
    training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                                      h, d_k, d_v, d_model, d_ff, n, dropout_rate)

    inputs = tf.keras.layers.Input(shape=(enc_seq_length,))
    outputs = training_model(inputs, training=True)
    outputs = K.max(outputs,axis=-1)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model2(seq_length=600, dropout_rate=0.1):
    
    # cnn
    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=(seq_length, 4), filters=128, kernel_size=8))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model


def model3(seq_length=600, dropout_rate=0.1):
    
    # embedding + cnn
    
    d_model = 16
    enc_vocab_size = 5 # Vocabulary size for the encoder
     
    inputs = tf.keras.layers.Input(shape=(seq_length,))
    outputs = Embedding(input_dim=enc_vocab_size, output_dim=d_model)(inputs)
    outputs = Conv1D(activation="relu", input_shape=(seq_length, d_model), filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model4(seq_length=600, dropout_rate=0.1):
    
    # transformer + cnn
    
    enc_vocab_size = 5 # Vocabulary size for the encoder
    dec_vocab_size = enc_vocab_size # Vocabulary size for the decoder

    enc_seq_length = seq_length  # Maximum length of the input sequence
    dec_seq_length = seq_length  # Maximum length of the target sequence

    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 32  # Dimensionality of the inner fully connected layer
    d_model = 16  # Dimensionality of the model sub-layers' outputs
    n = 1  # Number of layers in the encoder stack
    
    training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                                      h, d_k, d_v, d_model, d_ff, n, dropout_rate)

    inputs = tf.keras.layers.Input(shape=(enc_seq_length,))
    outputs = training_model(inputs, training=True)
    outputs = Conv1D(activation="relu", input_shape=(seq_length, d_model), filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8)(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model5(seq_length=600, dropout_rate=0.1):
    
    # cnn original
    
    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=(seq_length, 4), filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(activation="relu", filters=128, kernel_size=8, padding="same"))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model


def model6(seq_length=600, dropout_rate=0.1):
    
    # embedding + cnn original
    
    d_model = 16
    enc_vocab_size = 5 # Vocabulary size for the encoder
     
    inputs = tf.keras.layers.Input(shape=(seq_length,))
    outputs = Embedding(input_dim=enc_vocab_size, output_dim=d_model)(inputs)
    outputs = Conv1D(activation="relu", input_shape=(seq_length, d_model), filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model7(seq_length=600, dropout_rate=0.1):
    
    # embedding only
    
    d_model = 16
    enc_vocab_size = 5 # Vocabulary size for the encoder
     
    inputs = tf.keras.layers.Input(shape=(seq_length,))
    outputs = Embedding(input_dim=enc_vocab_size, output_dim=d_model)(inputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def model8(seq_length=600, dropout_rate=0.1):
    
    # transformer + cnn original
    
    enc_vocab_size = 5 # Vocabulary size for the encoder
    dec_vocab_size = enc_vocab_size # Vocabulary size for the decoder

    enc_seq_length = seq_length  # Maximum length of the input sequence
    dec_seq_length = seq_length  # Maximum length of the target sequence

    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 32  # Dimensionality of the inner fully connected layer
    d_model = 16  # Dimensionality of the model sub-layers' outputs
    n = 1  # Number of layers in the encoder stack
    
    training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                                      h, d_k, d_v, d_model, d_ff, n, dropout_rate)

    inputs = tf.keras.layers.Input(shape=(enc_seq_length,))
    outputs = training_model(inputs, training=True)
    outputs = Conv1D(activation="relu", input_shape=(seq_length, d_model), filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Conv1D(activation="relu", filters=128, kernel_size=8, padding="same")(outputs)
    outputs = MaxPooling1D()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model9(seq_length=600, dropout_rate=0.1):
    
    # transformer no max pooling
    enc_vocab_size = 5 # Vocabulary size for the encoder
    dec_vocab_size = enc_vocab_size # Vocabulary size for the decoder

    enc_seq_length = seq_length  # Maximum length of the input sequence
    dec_seq_length = seq_length  # Maximum length of the target sequence

    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 32  # Dimensionality of the inner fully connected layer
    d_model = 16  # Dimensionality of the model sub-layers' outputs
    n = 1  # Number of layers in the encoder stack
    
    word_embedding_layer = Embedding(input_dim=enc_vocab_size, output_dim=d_model)
    training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                                      h, d_k, d_v, d_model, d_ff, n, dropout_rate)

    inputs = tf.keras.layers.Input(shape=(enc_seq_length,))
    outputs = training_model(inputs, training=True)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def model10(seq_length=600, dropout_rate=0.1):
    
    # transformer, no max pooling, no masking
    enc_vocab_size = 5 # Vocabulary size for the encoder
    dec_vocab_size = enc_vocab_size # Vocabulary size for the decoder

    enc_seq_length = seq_length  # Maximum length of the input sequence
    dec_seq_length = seq_length  # Maximum length of the target sequence

    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 32  # Dimensionality of the inner fully connected layer
    d_model = 16  # Dimensionality of the model sub-layers' outputs
    n = 1  # Number of layers in the encoder stack
    
    word_embedding_layer = Embedding(input_dim=enc_vocab_size, output_dim=d_model)
    training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                                      h, d_k, d_v, d_model, d_ff, n, dropout_rate)

    inputs = tf.keras.layers.Input(shape=(enc_seq_length,))
    outputs = training_model(inputs, training=True, mask=False)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def model11(seq_length=600, dropout_rate=0.1):
    
    # embedding + LSTM
    d_model = 64
    enc_vocab_size = 5 # Vocabulary size for the encoder
    
    inputs = tf.keras.layers.Input(shape=(seq_length,))
    outputs = Embedding(input_dim=enc_vocab_size, output_dim=d_model)(inputs)
    outputs = LSTM(d_model)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def model12(seq_length=600, dropout_rate=0.1):
    
    # one-hot encoding, Graph Attention Networks
    
    n_node_features = 14
    n_edge_features = 3
    dropout = 0.1
    channels = 128
    l2_reg = 5e-4
    num_classes=1

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, 4))
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)

    graph_conv_1 = GATConv(channels,  attn_heads=4,
                             activation='relu',
                             kernel_regularizer=regularizers.l2(l2_reg),
                             use_bias=False)([X_in, fltr_in])

    graph_conv_2 = GATConv(64, attn_heads=4,
                             activation='relu',
                             use_bias=False)([graph_conv_1, fltr_in])

    #outputs = Flatten()(graph_conv_2)
    outputs = GlobalSumPool()(graph_conv_2)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model

def model13(seq_length=600, dropout_rate=0.1):
    
    # embedding, Graph Attention Networks
    
    n_node_features = 14
    n_edge_features = 3
    dropout = 0.1
    channels = 128
    l2_reg = 5e-4
    num_classes=1
    
    d_model = 16 # embedding dimention
    enc_vocab_size = 5 # Vocabulary size for the encoder

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, ))
    
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)
    
    X_out = Embedding(input_dim=enc_vocab_size, output_dim=d_model, input_length=seq_length)(X_in)
    
    graph_conv_1 = GATConv(channels,  attn_heads=4,
                             activation='relu',
                             kernel_regularizer=regularizers.l2(l2_reg),
                             use_bias=False)([X_out, fltr_in])

    graph_conv_2 = GATConv(64, attn_heads=4,
                             activation='relu',
                             use_bias=False)([graph_conv_1, fltr_in])

    #outputs = Flatten()(graph_conv_2)
    outputs = GlobalSumPool()(graph_conv_2)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model

def model14(seq_length=600, dropout_rate=0.1):
    
    # one-hot encoding, Graph convolution Networks
    
    n_node_features = 14
    n_edge_features = 3
    dropout = 0.1
    channels = 128
    l2_reg = 5e-4
    num_classes=1

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, 4))
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)

    graph_conv_1 = ARMAConv(channels)([X_in, fltr_in])

    graph_conv_2 = ARMAConv(64)([graph_conv_1, fltr_in])

    #outputs = Flatten()(graph_conv_2)
    outputs = GlobalSumPool()(graph_conv_2)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model


def model15(seq_length=600, dropout_rate=0.1):
    
    # embedding, Graph convolution Networks
    
    n_node_features = 14
    n_edge_features = 3
    dropout = 0.1
    channels = 128
    l2_reg = 5e-4
    num_classes=1
    
    d_model = 16 # embedding dimention
    enc_vocab_size = 5 # Vocabulary size for the encoder

    # Model definition
    X_in = tf.keras.layers.Input(shape=(seq_length, ))
    
    fltr_in = tf.keras.layers.Input((seq_length, seq_length), sparse=True)
    
    X_out = Embedding(input_dim=enc_vocab_size, output_dim=d_model, input_length=seq_length)(X_in)

    graph_conv_1 = ARMAConv(channels)([X_out, fltr_in])

    graph_conv_2 = ARMAConv(64)([graph_conv_1, fltr_in])

    #outputs = Flatten()(graph_conv_2)
    outputs = GlobalSumPool()(graph_conv_2)
    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)
    
    # Build model
    model = keras.Model(inputs=[X_in, fltr_in], outputs=outputs)
    
    return model
