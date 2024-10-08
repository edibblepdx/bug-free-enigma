# Ethan Dibble
# SVM classifier using CNN feature extraction

from FeatureExtract import FeatureExtract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pickle

class SVM:
    def __init__(self):
        self.clf = svm.SVC(kernel='rbf')

    def fit(self, x_train, y_train):
        """train SVM"""
        history = self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        """predict"""
        predictions = self.clf.predict(x_test)
        return predictions

    def save_model(self, file_path):
        """save model"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.clf, f)

    def load_model(self, file_path):
        """load model"""
        with open(file_path, 'rb') as f:
            self.clf = pickle.load(f)

def main():
    fe = FeatureExtract()
    fe.load_model('cnn6.keras')
    #x, y = fe.load_data('Data/genres_original', 'Data/features_30_sec.csv')
    x, y = fe.load_csv('mfccs.csv')

    # extract features using CNN
    features = fe.extract(x).numpy()
    print(type(features))
    print(features.shape)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(y)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

    # train SVM
    model = SVM()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    model.save_model('svm.pkl')

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

if __name__ == '__main__':
    main()