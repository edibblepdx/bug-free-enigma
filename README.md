```python
from FeatureExtract import FeatureExtract
```
Starting out
```python
fe = FeatureExtract()
x, y = fe.load_data('path/to/gtzan_wavs', 'path/to/gtzan_csv')
fe.train(features=x, labels=y, predict=False)
features = fe.extract(x)
```
Subsequent importing of the data can be made quicker with
```python
fe.save_csv('features.csv', features, labels)
x, y = fe.load_csv('features.csv')
```
To save and load the model after training
```python
fe.load_model(path)
fe.save_model(path=None, overwrite=False)
```
Example use with SVM
```python
fe = FeatureExtract()
fe.load_model('cnn5.keras')
x, y = fe.load_csv('features.csv')

features = fe.extract(x).numpy()
print(features.shape)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

clf = svm.SVC()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

# confusion matrix
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(y_test)
cm = confusion_matrix(true_labels, predicted_labels, labels=label_encoder.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
display.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title('Confusion Matrix')
plt.show()

# accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print (f"accuracy: {accuracy}")
```
