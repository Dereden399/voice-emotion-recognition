from keras import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.losses import SparseCategoricalCrossentropy

class CNN_model:
  def __init__(self):
    self.model = Sequential()
    self.model.add(Conv1D(256, 8, input_shape=(203, 1), activation="relu"))
    self.model.add(MaxPooling1D(2))
    self.model.add(Dense(128, activation="relu"))
    self.model.add(Flatten())
    self.model.add(Dense(8, activation="softmax"))
  
  def compile_model(self):
    self.model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy())

  def train_model(self, features, labels):
    self.model.fit(features.reshape(features.shape[0], features.shape[1], 1), labels, epochs=10)
  
  def predict_emotions(self, data_set):
    return self.model.predict(data_set.reshape(data_set.shape[0], data_set.shape[1], 1))