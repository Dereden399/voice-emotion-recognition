from keras import Sequential, models
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
import os

class CNN_model:
  def __init__(self):
    self.model = Sequential()
    self.model.add(Conv1D(256, 8, input_shape=(203, 1), activation="relu"))
    self.model.add(MaxPooling1D(2))
    self.model.add(Dense(128, activation="relu"))
    self.model.add(Flatten())
    self.model.add(Dense(8, activation="softmax"))
  
  def compile_model(self):
    if isinstance(self.model, Sequential):
      self.model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy())
    else:
      raise Exception("model is None")

  def train_model(self, features, labels, from_file=None):
    if from_file:
      if os.path.isfile(from_file) and from_file.endswith("cnn_model.keras"):
        self.model = models.load_model(from_file)
        return None;
    if isinstance(self.model, Sequential):
      self.model.fit(features.reshape(features.shape[0], features.shape[1], 1), labels, epochs=10)
      self.model.save("pickles/cnn_model.keras")
    else:
      raise Exception("model is None")
  
  def predict_emotions(self, data_set):
    if isinstance(self.model, Sequential):
      return self.model.predict(data_set.reshape(data_set.shape[0], data_set.shape[1], 1))
    else:
      raise Exception("model is None")