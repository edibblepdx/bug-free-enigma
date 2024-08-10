from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)

class APIError(Exception):
    """All Custom API Exceptions"""
    pass

class APIFileError(APIError):
    """Custom File Error Class"""
    code = 400
    description = 'file error'

def gen_spectrogram(file):
    # load the audio as a waveform y
    # store the sampling rate as sr
    target_sr = 22050

    # load wav file
    y, sr = librosa.load(file, sr=target_sr)

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
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
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
        spectrogram_img = gen_spectrogram(file)
        return send_file(spectrogram_img, mimetype='image/png', as_attachment=True, download_name='spectrogram.png')
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