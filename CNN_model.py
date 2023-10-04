from keras import Sequential
import keras.layers
from FeaturesExtractor import FeaturesExtractor

class CNN_model:
  data = FeaturesExtractor()
  is_trained = False
  X_train, X_test, y_train, y_test = ([], [], [], [])
  
  model = Sequential()