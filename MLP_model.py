from sklearn.neural_network import MLPClassifier

class MLP_model:
  model = MLPClassifier(max_iter=200, hidden_layer_sizes=[200, 200])
  
  def train_model(self, features, labels):
    try:
      self.model.fit(features, labels)
      return True
    except:
      return False
  def predict_emotions(self, data_set):
    return self.model.predict(data_set)
  
  

