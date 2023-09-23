from FeaturesExtractor import FeaturesExtractor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class MLP_model:
  data = FeaturesExtractor()
  model = MLPClassifier(max_iter=500, hidden_layer_sizes=[200, 200])
  is_trained = False
  X_train, X_test, y_train, y_test = ([], [], [], [])

  def load_model(self, pathToDataset, mode="pickle"):
    self.data.load_samples(pathToDataset, mode=mode)
    print("Completed")
  
  def train_model(self):
    if len(self.data.features) == 0:
      print("Load the model first")
    elif self.is_trained:
      print("The model is already trained")
    else:
      print("Fitting model...")
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.features, self.data.labels, test_size=0.33, shuffle=True)
      self.model.fit(self.X_train, self.y_train)
      print("Done")
  def predict(self, data_set):
    return self.model.predict(data_set)
  

