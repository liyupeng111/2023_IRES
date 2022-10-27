import tensorflow as tf
from tensorflow import keras, convert_to_tensor, string
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow import linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer, Embedding, MaxPooling1D
from tensorflow.keras.layers import LayerNormalization, ReLU, Dropout
from tensorflow.keras.layers import Activation, Flatten, Conv1D, BatchNormalization
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from keras import backend as K
from keras.backend import softmax

from focal_loss import BinaryFocalLoss
from functions import one_hot_encode_padding
from functions import model_cnn
from functions import LRScheduler, TransformerModel, encode_padding 
    
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