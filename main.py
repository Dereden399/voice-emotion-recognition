from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from EmotionRecognition import EmotionRecognition

system = EmotionRecognition("samples", mode="pickle")

system.train_models(load_saved=True)

mlp_validation_predicted = system.predict_mlp(system.X_val)
mlp_accuracy = accuracy_score(system.y_val, mlp_validation_predicted)
mlp_validation_predicted_prob = system.predict_mlp_proba(system.X_val)
mlp_loss = log_loss(system.y_val, mlp_validation_predicted_prob)
print(f"Acuracy for mlp model with validation set: {mlp_accuracy}")
print(f"Loss for mlp model with validation set: {mlp_loss}")

cnn_validation_predicted = system.predict_cnn(system.X_val)
cnn_accuracy = accuracy_score(system.y_val, cnn_validation_predicted)
cnn_validation_predicted_prob = system.predict_cnn_proba(system.X_val)
cnn_loss = log_loss(system.y_val, cnn_validation_predicted_prob)
print(f"Acuracy for cnn model with validation set: {cnn_accuracy}")
print(f"Loss for cnn model with validation set: {cnn_loss}")

if mlp_accuracy >= cnn_accuracy:
  print("MLP model gives better results for emotion recognition")
else:
  print("CNN model gives better results for emotion recognition")