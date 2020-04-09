from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import argparse
from tensorflow.keras.preprocessing import image
from Model import model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class FilePaths:
  test_dir = "/content/CatVsDog/data/dataset/test_set/"
  train_dir = "/content/CatVsDog/data/dataset/training_set/"

def train(train_dir, test_dir):

  data_generator = ImageDataGenerator(rescale = 1.0/255.0, zoom_range=0.2)

  batch_size = 32

  training_data = data_generator.flow_from_directory(directory = train_dir, target_size = (64,64), batch_size = batch_size, class_mode = 'binary')

  testing_data = data_generator.flow_from_directory(directory = test_dir, target_size = (64,64), batch_size = batch_size, class_mode = 'binary')

  mode = model(training_data)

  fitted_model = mode.fit_generator(training_data, steps_per_epoch = 1000, epochs = 20, validation_data = testing_data, validation_steps = 1000)

  mode.save('final_model.h5')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', help = 'train the NN', action = 'store_true')

  args = parser.parse_args()

  if args.train:
    train(FilePaths.train_dir, FilePaths.test_dir)

if __name__ == '__main__':
  main()
