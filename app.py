from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from base64 import encodebytes, b64encode
from FeatureExtract import FeatureExtract
from svm import SVM
import io

ALLOWED_EXTENSIONS = {'wav', 'mp3'}
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop','reggae', 'rock']
CNN_MODEL = './models/cnn6.keras'
SVM_MODEL = './models/svm6.pkl'

app = Flask(__name__)

class APIError(Exception):
    """All Custom API Exceptions"""
    pass

class APIFileError(APIError):
    """Custom File Error Class"""
    code = 400
    description = 'file error'

def gen_spectrogram(y, sr):
    """Generate a Spectrogram Image"""
    # load the audio as a waveform y
    # store the sampling rate as sr

    # Compute spectrogram
    S = librosa.stft(y)
    D = librosa.amplitude_to_db(abs(S), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Save to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='PNG')
    plt.close()
    img.seek(0)

    return img

def preprocess(y):
    """load and preprocess the audio file"""
    # extract MFCC (Mel Frequency Capstone Coefficients) features
    mfccs = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40) # returns np.ndarray [shape=(…, n_mfcc, t)]
    mfccs_scaled = np.mean(mfccs, axis=1) # calculate the mean over the time frames

    print("MFCCs scaled shape:", mfccs_scaled.shape)  # Debugging line
    #print("MFCCS raw:", mfccs)  # Debugging line
    #print("MFCCS scaled:", mfccs_scaled)  # Debugging line

    return mfccs_scaled.reshape(1, 40)

def extract_features(mfccs_scaled):
    """extract the features from the audio file"""
    fe = FeatureExtract()
    fe.load_model(CNN_MODEL)
    print(mfccs_scaled.shape)
    features = fe.extract(mfccs_scaled).numpy()

    #print("Extracted features shape:", features.shape)  # Debugging line
    #print("Extracted features", features)  # Debugging line

    return features.reshape(1, -1)

def predict(features):
    """attempt to classify the audio file""" 
    model = SVM()
    model.load_model(SVM_MODEL)
    prediction = model.predict(features)

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(LABELS)

    #print("Raw prediction:", prediction)  # Debugging line
    #print("Predicted label:", label_encoder.inverse_transform(prediction))  # Debugging line
    #print("Possible classes", label_encoder.classes_)  # Debugging line

    return label_encoder.inverse_transform(prediction)

def allowed_file(filename):
    """Validate file extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """homepage"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Uploaded file"""
    # check if the post request has the file part
    if 'file' not in request.files:
        raise APIFileError('No file part')
    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        raise APIFileError('No selected file')

    # operate on file
    if file and allowed_file(file.filename):
        # load wav file
        y, sr = librosa.load(file, sr=22050)

        # spectrogram image
        spectrogram_img = gen_spectrogram(y, sr)

        # classify
        mfccs_scaled = preprocess(y)
        features = extract_features(mfccs_scaled)
        prediction = predict(features)
        print(prediction[0])
        
        # return send_file(spectrogram_img, mimetype='image/png', as_attachment=True, download_name='spectrogram.png')

        # encode image to base64
        spectrogram_img_encoded = b64encode(spectrogram_img.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': prediction[0]
            , 'spectrogram': spectrogram_img_encoded
        })

    else:
        raise APIFileError('Invalid extension')

@app.errorhandler(APIError)
def handle_exception(err):
    """Return custom JSON when APIError or its children are raised"""
    response = {"error": err.description, "message": ""}
    if len(err.args) > 0:
        response["message"] = err.args[0]
    return jsonify(response), err.code

if __name__ == '__main__':
    app.run(debug=True)
