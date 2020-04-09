from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import tensorflow as tf

def load_image(filename):
  img = image.load_img(filename, target_size = (64,64))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis = 0)

  return img

def predict(path):
  img = load_image(path)
  model = tf.keras.models.load_model('model/final_model.h5')
  result = model.predict(x = img)
  print(result)
  if result[0][0] == 1:
    prediction = 'Dog'
  else:
    prediction = 'Cat'
  print(prediction)

  return prediction

def derive_by_web(path):
  prediction = predict(path)

  return prediction