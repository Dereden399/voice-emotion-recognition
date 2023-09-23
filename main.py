from MLP_model import MLP_model
from sklearn.metrics import accuracy_score, log_loss

model = MLP_model()

model.load_model("samples", mode="pickle")
print(model.data.features.shape)
model.train_model()


predicted = model.predict(model.X_test)
accuracy = accuracy_score(model.y_test, predicted)
test_predicted_prob = model.model.predict_proba(model.X_test)
loss = log_loss(model.y_test, test_predicted_prob)
print(f"Acuracy for test set: {accuracy}")
print(f"Loss for test set: {loss}")