from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'data/classmate/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({
        "message": "File uploaded successfully", 
        "url": f"http://localhost:8080/audio/{file.filename}"
    }), 200

@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    """Serve audio files from the upload folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # Run the Flask app on all network interfaces, port 8080
    app.run(host='0.0.0.0', port=8080)
