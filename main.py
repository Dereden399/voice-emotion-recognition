from MLP_model import MLP_model
from sklearn.metrics import accuracy_score

model = MLP_model()

model.load_model("samples", mode="new")
print(model.data.features.shape)
model.train_model()

train_predicted =  model.predict(model.X_train)
train_accuracy = accuracy_score(model.y_train, train_predicted)
print(f"Acuracy for training set: {train_accuracy}")

predicted = model.predict(model.X_test)
accuracy = accuracy_score(model.y_test, predicted)
print(f"Acuracy for test set: {accuracy}")