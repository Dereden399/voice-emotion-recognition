from MLP_model import MLP_model
from sklearn.metrics import accuracy_score, log_loss
from keras import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import keras
from FeaturesExtractor import FeaturesExtractor
from sklearn.model_selection import train_test_split
import numpy as np
from EmotionRecognition import EmotionRecognition

"""model = MLP_model()

model.load_model("samples", mode="pickle")
print(model.data.features.shape)
model.train_model()


predicted = model.predictSet(model.X_test)
accuracy = accuracy_score(model.y_test, predicted)
test_predicted_prob = model.model.predict_proba(model.X_test)
loss = log_loss(model.y_test, test_predicted_prob)
print(f"Acuracy for test set: {accuracy}")
print(f"Loss for test set: {loss}")

predicted = model.predictEmotion("test_samples/test_sample.wav")
print(predicted)"""


"""data = FeaturesExtractor()
data.load_samples("samples/", mode="pickle")
data.labels = data.labels - 1

newFeatures = data.features.reshape(data.features.shape[0], data.features.shape[1], 1)
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(newFeatures, data.labels, test_size=0.33, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.33, shuffle=True)

print(X_train.shape, X_val.shape, X_test.shape)

model = Sequential()
model.add(Conv1D(256, 8, input_shape=(203, 1), activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(8, activation="softmax"))

model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy())

model.fit(X_train, y_train, epochs=10)

y_predicted = np.argmax(model.predict(X_val), axis=1)
print(y_predicted.shape)
print(f"Accuracy: {accuracy_score(y_val, y_predicted)}")"""

system = EmotionRecognition("samples", mode="pickle")

system.train_models()

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