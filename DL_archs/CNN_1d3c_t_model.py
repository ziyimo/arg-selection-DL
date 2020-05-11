
## Required layers ##
# from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation
# from tensorflow.keras.layers import Conv2D, AveragePooling2D

def CNN_1d3c_t():
  K = 100 # of time points
  W = 13 # of windows
  In = Input(shape=(W*3, K, 1))

  x = Conv2D(filters=256, kernel_size=(13, 2), strides=(13, 1), activation='relu')(In)
  #x = AveragePooling2D(pool_size=(2, 1))(x)
  x = Conv2D(filters=256, kernel_size=(3, 2), strides=1, use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Flatten()(x)

  x = Dense(256, activation='relu')(x)
  x = Dropout(0.1)(x)
  x = Dense(256, use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Dense(128, activation='relu')(x)
  x = Dense(64, activation='relu')(x)

  Out = Dense(1, activation='linear')(x)

  CNN1d3c = Model(inputs=In, outputs=Out)
  CNN1d3c.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return CNN1d3c