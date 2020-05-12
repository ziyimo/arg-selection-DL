
## Required layers ##
# from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation
# from tensorflow.keras.layers import LSTM, Bidirectional

def bidirectional_LSTM():
  K = 100 # of time points
  W = 13 # of windows
  In = Input(shape=(W, 3*K))

  x = Bidirectional(LSTM(256, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(In)
  x = Bidirectional(LSTM(128, recurrent_dropout=0.1))(x)
  x = Dense(128, use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Dense(64, activation="relu")(x)

  Out = Dense(1, activation='linear')(x)

  biLSTM = Model(inputs=In, outputs=Out)
  biLSTM.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return biLSTM