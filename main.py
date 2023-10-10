from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from EmotionRecognition import EmotionRecognition
import matplotlib.pyplot as plt
import seaborn as sns



system = EmotionRecognition("samples", mode="pickle")

system.train_models(load_saved=False)

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

mlp_test_predicted = system.predict_mlp(system.X_test)
mlp_test_accuracy = accuracy_score(system.y_test, mlp_test_predicted)
mlp_test_loss = log_loss(system.y_test, system.predict_mlp_proba(system.X_test))

matrix = confusion_matrix(system.y_test, mlp_test_predicted)
ax= plt.subplot()

sns.heatmap(matrix, annot=True, fmt='g', ax=ax)

ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
ax.set_title(f'Accuracy: {(mlp_test_accuracy*100):.1f}%\nLoss: {mlp_test_loss:.2f}',fontsize=15)

plt.show()

