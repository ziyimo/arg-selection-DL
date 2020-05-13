
## Required layers ##
# from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation
# from tensorflow.keras.layers import LSTM, Bidirectional

def LSTM_t():
  K = 100 # of time points
  W = 13 # of windows
  In = Input(shape=(K, 3*W))

  x = LSTM(256, dropout=0, recurrent_dropout=0, return_sequences=True)(In)
          #kernel_regularizer=l2(l=1e-6), recurrent_regularizer=l2(l=1e-6),
  
  x = LSTM(128, recurrent_dropout=0)(x) #kernel_regularizer=l2(l=1e-6), recurrent_regularizer=l2(l=1e-6),
  
  x = Dense(128, use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Dense(64, activation="relu")(x)

  Out = Dense(1, activation='linear')(x)

  LSTM_t = Model(inputs=In, outputs=Out)
  LSTM_t.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return LSTM_t