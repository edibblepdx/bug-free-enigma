# Ethan Dibble
# SVM classifier using CNN feature extraction

from FeatureExtract import FeatureExtract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

def main():
    fe = FeatureExtract()
    fe.load('cnn5.keras')
    #x, y = fe.load_data('Data/genres_original', 'Data/features_30_sec.csv')
    x, y = fe.load_csv('features.csv')

    #print(fe.model.summary)
    #print(fe.feature_extractor.summary)

    features = fe.extract(x).numpy()
    print(type(features))
    print(features.shape)
    #features = features.reshape(999,-1)
    print(features.shape)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(y)
    #labels_onehot = to_categorical(labels_encoded)

    x_train, x_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    #predicted_classes = np.argmax(predictions, axis=1) # largest indices
    #predicted_labels = label_encoder.inverse_transform(predicted_classes)
    #true_classes = np.argmax(y_test, axis=1)
    #true_labels = label_encoder.inverse_transform(true_classes)

    print(np.unique(y_test))
    print(label_encoder.classes_)

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