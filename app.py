from flask import Flask, request, jsonify, send_file
import os
import uuid
import numpy as np
from onnxruntime import InferenceSession
import scipy.io.wavfile as wavfile

app = Flask(__name__)

# Adjust these paths as needed.
# In many add-on setups, you can mount your Home Assistant config directory (or a subfolder) so these files are accessible.
ONNX_MODEL_PATH = "/config/onnx/model.onnx"     # Place your ONNX model in /config/onnx/
VOICES_PATH = "/config/voices/af.bin"             # Place your voices file in /config/voices/
OUTPUT_DIR = "/config/www/kokoro_tts"             # The add-on will write the audio here

def text_to_tokens(text: str) -> list:
    """
    Convert input text to token IDs.
    This is a placeholder; in practice, convert text to phonemes then to token IDs.
    """
    # Example fixed token list (replace with your own conversion)
    tokens = [
        50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16,
        102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83,
        54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53,
        16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158,
        123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102,
        54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156,
        43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156,
        51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4
    ]
    return tokens

def generate_tts_audio(text: str) -> str:
    tokens = text_to_tokens(text)
    if len(tokens) > 510:
        raise ValueError("Token list is too long")

    # Load the voices file (assumes it contains float32 values with shape (N, 1, 256))
    voices = np.fromfile(VOICES_PATH, dtype=np.float32).reshape(-1, 1, 256)
    ref_s = voices[len(tokens)]
    
    # Pad tokens (context length is 512, so reserve two spots)
    tokens = [[0] + tokens + [0]]
    
    # Load the ONNX model
    sess = InferenceSession(ONNX_MODEL_PATH)
    model_inputs = {
        "input_ids": np.array(tokens, dtype=np.int64),
        "style": ref_s,
        "speed": np.ones(1, dtype=np.float32)
    }
    outputs = sess.run(None, model_inputs)
    audio = outputs[0]
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Generate a unique filename
    filename = f"tts_{uuid.uuid4().hex}.wav"
    output_file = os.path.join(OUTPUT_DIR, filename)
    # Write the audio file (sample rate: 24000)
    wavfile.write(output_file, 24000, audio[0])
    return output_file

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text']
    try:
        audio_file = generate_tts_audio(text)
        # Return the file as an attachment
        return send_file(audio_file, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Listen on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000)
