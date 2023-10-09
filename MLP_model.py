from sklearn.neural_network import MLPClassifier
import skops.io as sio
import os

class MLP_model:
  model = MLPClassifier(max_iter=200, hidden_layer_sizes=[200, 200])
  
  def train_model(self, features, labels, from_file=None):
    if from_file:
      if os.path.isfile(from_file) and from_file.endswith("mlp_model.skop"):
        unknown_types = sio.get_untrusted_types(file="my-model.skops")
        self.model = sio.load("my-model.skops", trusted=unknown_types)
        return True
    try:
      self.model.fit(features, labels)
      sio.dump(self.model, "pickles/mlp_model.skop")
      return True
    except:
      return False

  def predict_emotions(self, data_set):
    return self.model.predict(data_set)
  
  

