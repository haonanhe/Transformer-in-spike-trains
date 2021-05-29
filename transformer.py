import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
import numpy as np

# Sinusoidal Position Encoding
class TriPositionEncoding(layers.Layer):
    def __init__(self, maxlen, model_size):
        super(TriPositionEncoding, self).__init__()
        self.maxlen = maxlen
        self.model_size = model_size
      
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'model_size': self.model_size,
        })
        return config 
    
    def call(self, inputs):
        PE = np.zeros((self.maxlen, self.model_size))
        for i in range(self.maxlen):
            for j in range(self.model_size):
                if j % 2 == 0:
                    PE[i, j] = np.sin(i / 10000 ** (j / self.model_size))
                else:
                    PE[i, j] = np.cos(i / 10000 ** ((j-1) / self.model_size))
        PE = tf.constant(PE, dtype=tf.float32)
        return inputs + PE

# Learned Position Encoding
class PositionEncoding(layers.Layer):
    def __init__(self, maxlen, model_size):
        super(PositionEncoding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=model_size)
        self.maxlen = maxlen
        self.model_size = model_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pos_emb': self.pos_emb,
            'maxlen': self.maxlen,
            'model_size': self.model_size
        })
        return config 
        
    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

# Transformer Unit
class Transformer(layers.Layer):
    def __init__(self, embed_dim, n_heads, rate=0.1, atten_axes=1):
        super(Transformer, self).__init__()
        self.attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim, attention_axes=atten_axes)
        self.ffn = keras.Sequential(
            [layers.Dense(4*embed_dim, activation='relu'), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attn': self.attn,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config 
    
    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) #residual

# Stacked Transformer Model 
class TransformerDecoder(object):
    def __init__(self, num_layers=1, num_epochs=10, dropout=0.1, verbose=0, maxlen=13, num_heads=2, mode='Temporal'):
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.verbose = verbose
        self.maxlen = maxlen
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mode = mode

    def fit(self, x_train, y_train, x_valid, y_valid, callbacks_list): # x:[n_samples,n_time_bins,n_neurons]
        # Hidden layer
        model = Sequential()
        if self.mode == 'Temporal':
          model.add(TriPositionEncoding(maxlen=x_train.shape[1], model_size=x_train.shape[2]))
          for i in range(self.num_layers):
            model.add(Transformer(embed_dim=x_train.shape[2], n_heads=self.num_heads, rate=self.dropout, atten_axes=1))
        elif self.mode == 'Spatial':
          model.add(layers.Reshape((x_train.shape[2], x_train.shape[1]), input_shape=(x_train.shape[1],x_train.shape[2])))
          # To test if there's some positional patterns among neurons (spatial).
          # model.add(PositionEncoding(maxlen=x_train.shape[2], model_size=x_train.shape[1]))
          for i in range(self.num_layers):
            model.add(Transformer(embed_dim=x_train.shape[1], n_heads=self.num_heads, rate=self.dropout, atten_axes=1))
        elif self.mode == 'SpatialTemporal':
          # Spatial
          model.add(layers.Reshape((x_train.shape[2], x_train.shape[1]), input_shape=(x_train.shape[1],x_train.shape[2])))
          model.add(Transformer(embed_dim=x_train.shape[1], n_heads=self.num_heads, rate=self.dropout, atten_axes=1))
          # Temporal
          model.add(layers.Reshape((x_train.shape[1], x_train.shape[2]), input_shape=(x_train.shape[2],x_train.shape[1])))
          model.add(TriPositionEncoding(maxlen=x_train.shape[1], model_size=x_train.shape[2]))
          model.add(Transformer(embed_dim=x_train.shape[2], n_heads=self.num_heads, rate=self.dropout, atten_axes=1))
        # Output layer
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(y_train.shape[1]))

        # Train & Val
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])  # metrics：evaluation criteria
        history = model.fit(x_train, y_train, epochs=self.num_epochs, verbose=self.verbose, validation_data=(x_valid, y_valid), callbacks=callbacks_list)
        self.model = model
    
    def predict(self, x_test):
        # Test
        y_test_preds = self.model.predict(x_test)
        return y_test_preds

class CNNTransDecoder(object):
    def __init__(self, chans=32, num_epochs=10, dropout=0.1, verbose=0, maxlen=13, num_heads=2, mode='ST'):
        self.chans = chans
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.verbose = verbose
        self.maxlen = maxlen
        self.num_heads = num_heads 
        self.mode = mode

    def fit(self, x_train, y_train, x_valid, y_valid, callbacks_list): #x:[n_samples,n_time_bins,n_neurons]
        # Hidden layer
        model = Sequential()
        if self.mode == 'ST':
          # Spatial CNN
          model.add(layers.Reshape((x_train.shape[1], x_train.shape[2], 1), input_shape=(x_train.shape[1],x_train.shape[2])))
          model.add(layers.Conv2D(self.chans, (1, x_train.shape[2])))
          model.add(layers.BatchNormalization(axis = -1))
          model.add(layers.Activation('relu'))
          model.add(layers.Dropout(self.dropout))
          # Temporal Trans
          model.add(layers.Reshape((x_train.shape[1], self.chans), input_shape=(x_train.shape[1],1,self.chans)))
          model.add(TriPositionEncoding(maxlen=x_train.shape[1], model_size=self.chans))
          model.add(Transformer(embed_dim=self.chans, n_heads=self.num_heads, rate=self.dropout, atten_axes=1))
          model.add(layers.core.Flatten())
          model.add(layers.Dense(y_train.shape[1]))
        elif self.mode == 'TS':
          # Temporal CNN
          model.add(layers.Reshape((x_train.shape[1], x_train.shape[2], 1), input_shape=(x_train.shape[1],x_train.shape[2])))
          model.add(layers.Conv2D(self.chans, (5, 1)))
          model.add(layers.BatchNormalization(axis = -1))
          model.add(layers.Activation('relu'))
          model.add(layers.Dropout(self.dropout))
          # Spatial Trans
          model.add(layers.Reshape((self.chans*6, x_train.shape[2]), input_shape=(6, x_train.shape[2], self.chans)))
          model.add(layers.AveragePooling1D(6))
          model.add(TriPositionEncoding(maxlen=self.chans, model_size=x_train.shape[2]))
          model.add(Transformer(embed_dim=x_train.shape[2], n_heads=self.num_heads, rate=self.dropout, atten_axes=1))
          model.add(layers.core.Flatten())
          model.add(layers.Dense(y_train.shape[1]))
        elif self.mode == 'STT':
          # SpatialTemporal CNN
          model.add(layers.Reshape((x_train.shape[1], x_train.shape[2], 1), input_shape=(x_train.shape[1],x_train.shape[2])))
          model.add(layers.Conv2D(self.chans, (1, x_train.shape[2])))#Spatial
          model.add(layers.Conv2D(self.chans, (3, 1), padding='same'))#Temporal
          model.add(layers.BatchNormalization(axis = -1))
          model.add(layers.Activation('relu'))
          model.add(layers.Dropout(self.dropout))
          
          # An extra spatial operation test
          # # Spatial Trans
          # model.add(layers.Reshape((self.chans, x_train.shape[1]), input_shape=(x_train.shape[1],1,self.chans)))
          # model.add(Transformer(embed_dim=x_train.shape[1], n_heads=self.num_heads, rate=self.dropout, atten_axes=1))
          
          # Temporal Trans
          # model.add(layers.Reshape((x_train.shape[1], self.chans), input_shape=(self.chans, x_train.shape[1])))
          model.add(TriPositionEncoding(maxlen=x_train.shape[1], model_size=self.chans))
          model.add(Transformer(embed_dim=self.chans, n_heads=self.num_heads, rate=self.dropout, atten_axes=1))

        # Output
        model.add(layers.core.Flatten())
        model.add(layers.Dense(y_train.shape[1]))

        # Train & Val
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])  # metrics：evaluation criteria
        history = model.fit(x_train, y_train, epochs=self.num_epochs, verbose=self.verbose, validation_data=(x_valid, y_valid), callbacks=callbacks_list)
        self.model = model
    
    def predict(self, x_test):
        # Test
        y_test_preds = self.model.predict(x_test)
        return y_test_preds

# Try to combine attention with GRU
class AttenGRU(layers.Layer):
    def __init__(self, embed_dim, n_heads, n_units, rate=0.1, in_shape=(1,1)):
        super(AttenGRU, self).__init__()
        self.attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim)
        self.gru = layers.GRU(n_units, input_shape=in_shape, dropout=rate, recurrent_dropout=rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attn': self.attn,
            'gru': self.gru,
            'layernorm1': self.layernorm1,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config 
    
    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        out2 = self.gru(out1)
        out2 = self.dropout2(out2, training=training)
        return  out2

class AttenGRUDecoder(object):
    def __init__(self, num_units=400, num_epochs=10, dropout=0.1, verbose=0, maxlen=13, num_heads=2):
        self.num_units = num_units
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.verbose = verbose
        self.maxlen = maxlen
        self.num_heads = num_heads 

    def fit(self, x_train, y_train, x_valid, y_valid, callbacks_list): #x:[n_samples,n_time_bins,n_neurons]
        # Hidden layer
        model = Sequential()
        model.add(TriPositionEncoding(maxlen=x_train.shape[1], model_size=x_train.shape[2]))
        model.add(AttenGRU(embed_dim=x_train.shape[2], n_heads=self.num_heads, n_units=self.num_units, rate=self.dropout, in_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(layers.Dense(y_train.shape[1]))

        # Train & Val
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])  # metrics：evaluation criteria
        history = model.fit(x_train, y_train, epochs=self.num_epochs, verbose=self.verbose, validation_data=(x_valid, y_valid), callbacks=callbacks_list)
        self.model = model
    
    def predict(self, x_test):
        # Test
        y_test_preds = self.model.predict(x_test)
        return y_test_preds

class TransGRUDecoder(object):
    def __init__(self, num_units=400, num_epochs=10, dropout=0.1, verbose=0, maxlen=13, num_heads=2):
        self.num_units = num_units
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.verbose = verbose
        self.maxlen = maxlen
        self.num_heads = num_heads 

    def fit(self, x_train, y_train, x_valid, y_valid, callbacks_list): #x:[n_samples,n_time_bins,n_neurons]
        # Hidden layer
        model = Sequential()
        model.add(TriPositionEncoding(maxlen=x_train.shape[1], model_size=52))
        model.add(Transformer(embed_dim=52, n_heads=8, rate=self.dropout, atten_axes=1))
        model.add(layers.GRU(self.num_units, dropout=self.dropout, recurrent_dropout=self.dropout, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(layers.Dense(y_train.shape[1]))

        # Train & Val
        model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])  # metrics：evaluation criteria
        history = model.fit(x_train, y_train, epochs=self.num_epochs, verbose=self.verbose, validation_data=(x_valid, y_valid), callbacks=callbacks_list)
        self.model = model
    
    def predict(self, x_test):
        # Test
        y_test_preds = self.model.predict(x_test)
        return y_test_preds