from FeaturesExtractor import FeaturesExtractor
from MLP_model import MLP_model
from CNN_model import CNN_model
from sklearn.model_selection import train_test_split
from yaspin import yaspin
from yaspin.spinners import Spinners
import numpy as np

class EmotionRecognition:

  def __init__(self, pathToDataset, mode="pickle"):
    self.data = FeaturesExtractor()
    self.mlp_model = MLP_model()
    self.cnn_model = CNN_model()
    self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = ([], [], [], [], [], [])
    if mode == "pickle":
      with yaspin(Spinners.line, text="Loading saved features...") as sp:
        success = self.data.load_samples(pathToDataset, mode=mode)
        if success:
          sp.ok("DONE!")
        else:
          sp.fail("FAIL!")
    else:
      self.data.load_samples(pathToDataset, mode=mode)
    with yaspin(Spinners.line, text="Splitting features into sets...") as sp:
      self.X_train, X_val_and_test, self.y_train, y_val_and_test = train_test_split(self.data.features, self.data.labels, test_size=0.33, shuffle=True)
      self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.33, shuffle=True)
      sp.ok("DONE!")
    with yaspin(Spinners.line, text="Compiling CNN model...") as sp:
      self.cnn_model.compile_model()
      sp.ok("DONE!")
  
  def train_models(self):
    with yaspin(Spinners.line, text="Training MLP model...") as sp:
      success = self.mlp_model.train_model(self.X_train, self.y_train)
      if success:
        sp.ok("DONE!")
      else:
        sp.fail("FAIL!")
    self.cnn_model.train_model(self.X_train, self.y_train)

  def predict_mlp(self, datapoints):
    return self.mlp_model.predict_emotions(datapoints)
  
  def predict_mlp_proba(self, datapoints):
    return self.mlp_model.model.predict_proba(datapoints)
  
  def predict_cnn(self, datapoints):
    return np.argmax(self.cnn_model.predict_emotions(datapoints), axis=1)
  
  def predict_cnn_proba(self, datapoints):
    return self.cnn_model.predict_emotions(datapoints)

