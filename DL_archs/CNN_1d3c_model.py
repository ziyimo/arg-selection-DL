## Required layers ##
# from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D

def CNN_1d3c():
  K = 100 # of time points
  W = 13 # of windows
  In = Input(shape=(W, K*3, 1))

  x = Conv2D(filters=256, kernel_size=(2, 100), strides=(1, 100), activation='relu')(In)
  x = Conv2D(filters=256, kernel_size=(2, 3), strides=1, activation='relu')(x)
  #x = AveragePooling2D(pool_size=(2, 1))(x)
  x = Flatten()(x)

  x = Dense(256, activation='relu')(x)
  #x = Dropout(0.2)(x)
  x = Dense(256, activation='relu')(x)
  #x = Dropout(0.2)(x)
  x = Dense(256, activation='relu')(x)

  Out = Dense(1, activation='linear')(x)

  CNN1d3c = Model(inputs=In, outputs=Out)
  CNN1d3c.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return CNN1d3c