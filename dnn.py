from FeatureExtract import FeatureExtract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import models, layers, utils
import matplotlib.pyplot as plt
import numpy as np

fe = FeatureExtract()
fe.load_model('cnn6.keras')
x, y = fe.load_csv('features.csv')

# extract features using CNN
features = fe.extract(x).numpy()
print(type(features))
print(features.shape)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(y)
labels_onehot = utils.to_categorical(labels_encoded)

# train test split
x_train, x_test, y_train, y_test = train_test_split(features
                                                    , labels_onehot
                                                    , test_size=0.2
                                                    , random_state=42
                                                    , stratify=labels_onehot)

#Creating the model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam'
              , loss='categorical_crossentropy'
              , metrics=['accuracy'])
              
# training the model
history = model.fit(x_train, y_train, epochs=20, batch_size=128)

# predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)
true_classes = np.argmax(y_test, axis=1)
true_labels = label_encoder.inverse_transform(true_classes)

# confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=label_encoder.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
display.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title('Confusion Matrix')
plt.show()

# accuracy
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print (f"accuracy: {accuracy}")

# Plot training accuracy and loss over epochs
plt.figure(figsize=(8, 6))

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training Accuracy and Loss over Epochs')
plt.legend()
plt.xticks(np.arange(0, len(history.history['accuracy']), step=5))

plt.show()