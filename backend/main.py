from flask import Flask, request, jsonify, send_from_directory
from transcript.chatbot import Chatbot
from sqlalchemy import create_engine, text
import os
from passlib.hash import pbkdf2_sha256

app = Flask(__name__)

# Database configuration
DATABASE_URI = "postgresql://admin:secret@localhost:5434/testdb"
engine = create_engine(DATABASE_URI)

# Initialize Chatbot instance
chatbot = Chatbot()

# Global upload folder (base for user-specific folders)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploaded_audio")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Expects JSON with 'question' and optional 'conversation_history' fields.
    Calls the chatbot to generate a response using the provided conversation history.
    """
    data = request.get_json()
    question = data.get("question")
    conversation_history = data.get("conversation_history", [])
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        response = chatbot.generate_response(question, engine, conversation_history)
        return jsonify({"answer": response.get("chatbot_response")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    """
    Handles audio file uploads.
    Expects a file in the request and a 'username' in the form data.
    Saves the file in the user's specific storage folder and returns a URL.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get the username from form data
    username = request.form.get("username")
    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Retrieve the user's storage folder from the database
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT storage_folder FROM users WHERE username = :username"),
            {"username": username}
        )
        row = result.fetchone()
        if not row:
            return jsonify({"error": "User not found"}), 404
        user_folder = row[0]

    # Create the user-specific directory if it doesn't exist
    user_directory = os.path.join(UPLOAD_FOLDER, user_folder)
    os.makedirs(user_directory, exist_ok=True)

    # Save the file in the user's folder
    file_path = os.path.join(user_directory, file.filename)
    file.save(file_path)

    # Construct a URL to serve the uploaded file (user-specific route)
    file_url = request.host_url + f"audio/{user_folder}/{file.filename}"
    return jsonify({"url": file_url}), 200

@app.route('/audio/<folder>/<filename>', methods=['GET'])
def get_user_audio(folder, filename):
    """
    Serves audio files from a user-specific folder.
    """
    directory = os.path.join(UPLOAD_FOLDER, folder)
    return send_from_directory(directory, filename)

# ------------------------------
# Endpoints for Login/Registration
# ------------------------------

@app.route('/api/register', methods=['POST'])
def register():
    """
    Registers a new user.
    Expects JSON with 'username' and 'password'.
    """
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    # Hash the password
    hashed_password = pbkdf2_sha256.hash(password)

    try:
        with engine.begin() as conn:
            # Insert new user into users table
            conn.execute(text("""
                INSERT INTO users (username, password, storage_folder, question_count)
                VALUES (:username, :password, :storage_folder, :question_count)
            """), {
                "username": username,
                "password": hashed_password,
                "storage_folder": f"user_{username}",
                "question_count": 0
            })
        return jsonify({"success": True, "message": "Registration successful"}), 200
    except Exception as e:
        # If username already exists or another error occurs
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
    """
    Logs in an existing user.
    Expects JSON with 'username' and 'password'.
    """
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT user_id, username, password, storage_folder, question_count 
                FROM users WHERE username = :username
            """), {"username": username})
            user = result.fetchone()
            if user and pbkdf2_sha256.verify(password, user.password):
                return jsonify({
                    "success": True,
                    "user": {
                        "user_id": user.user_id,
                        "username": user.username,
                        "storage_folder": user.storage_folder,
                        "question_count": user.question_count
                    }
                }), 200
            else:
                return jsonify({"success": False, "message": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ------------------------------
# List User Audios
# ------------------------------

@app.route('/api/list-audios', methods=['POST'])
def list_audios():
    """
    Returns a list of uploaded audios for a user.
    Expects JSON with 'username'.
    """
    data = request.get_json()
    username = data.get("username")
    if not username:
        return jsonify({"error": "Username required"}), 400

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT storage_folder FROM users WHERE username = :username"),
            {"username": username}
        )
        row = result.fetchone()
        if not row:
            return jsonify({"error": "User not found"}), 404
        user_folder = row[0]

    user_directory = os.path.join(UPLOAD_FOLDER, user_folder)
    if not os.path.exists(user_directory):
        return jsonify({"audios": []}), 200

    file_list = os.listdir(user_directory)
    audios = []
    for filename in file_list:
        file_url = request.host_url + f"audio/{user_folder}/{filename}"
        audios.append({"filename": filename, "url": file_url})
    return jsonify({"audios": audios}), 200

# ------------------------------
# Delete Audio
# ------------------------------

@app.route('/api/delete-audio', methods=['POST'])
def delete_audio():
    """
    Deletes an audio file for a user.
    Expects JSON with 'username' and 'filename'.
    """
    data = request.get_json()
    username = data.get("username")
    filename = data.get("filename")
    if not username or not filename:
        return jsonify({"error": "Username and filename required"}), 400

    # Retrieve user's storage folder
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT storage_folder FROM users WHERE username = :username"),
            {"username": username}
        )
        row = result.fetchone()
        if not row:
            return jsonify({"error": "User not found"}), 404
        user_folder = row[0]

    user_directory = os.path.join(UPLOAD_FOLDER, user_folder)
    file_path = os.path.join(user_directory, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(file_path)
        return jsonify({"success": True, "message": "File deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
