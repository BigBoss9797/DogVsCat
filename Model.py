from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def model(training_data):
  model = Sequential()
  model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training_data.image_shape))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.3))

  model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.2))

  model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.15))

  model.add(Flatten())
  model.add(Dense(units = 32, activation = 'relu'))
  model.add(Dropout(rate = 0.15))

  model.add(Dense(units = 64, activation = 'relu'))
  model.add(Dropout(rate = 0.1))

  model.add(Dense(units = len(set(training_data.classes)), activation = 'softmax'))
  model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

  model.summary()

  return model
