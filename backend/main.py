from flask import Flask, request, jsonify, send_from_directory
from transcript.chatbot import Chatbot
from sqlalchemy import create_engine, text
import os, platform, threading, torch, requests, time
from passlib.hash import pbkdf2_sha256
import subprocess

app = Flask(__name__)

# ———————— Hostname & DB config ————————
computer_name = platform.node().lower()
if computer_name == "desktop-jacor":
    DATABASE_URI = "postgresql://admin:secret@localhost:5432/testdb"
    LLM_MODEL    = "cogito:14b" 
else:
    DATABASE_URI = "postgresql://admin:secret@localhost:5434/testdb"
    LLM_MODEL    = "deepseek-r1:7b"

engine      = create_engine(DATABASE_URI)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploaded_audio")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ———————— Globals & Lock ————————
chatbot      = None
_load_lock   = threading.Lock()

# How long to keep the models in memory after last request (seconds)
IDLE_UNLOAD_DELAY = 5 *60 
_unload_timer  = None


def _do_unload():
    """Called by timer: actually free GPU and delete chatbot."""
    global chatbot
    with _load_lock:
        if chatbot:
            # free PyTorch GPU cache
            torch.cuda.empty_cache()
            # tell Ollama server to unload the model
            try:
                requests.post(
                    "http://localhost:11434/api/stop",
                    json={"name": LLM_MODEL},
                    timeout=5 # caps how long the client will wait for a response before giving up
                )
            except requests.exceptions.RequestException as e:
                print(f"Error unloading model: {e}")
            # delete the chatbot instance
            chatbot = None

# unload model AFTER IDLE_UNLOAD_DELAY seconds of INACTIVITY
def _schedule_unload():
    """Restart the unload timer."""
    global _unload_timer
    # If there’s already a pending unload timer, cancel it...
    if _unload_timer:
        _unload_timer.cancel()
    _unload_timer = threading.Timer(IDLE_UNLOAD_DELAY, _do_unload)
    _unload_timer.daemon = True
    _unload_timer.start()


@app.route("/chat", methods=["POST"])
def chat():
    global chatbot

    data = request.get_json() or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # ——— 1️⃣ Lazy initialize under lock ———
    with _load_lock:
        if chatbot is None:
            chatbot = Chatbot(
                llm_model=LLM_MODEL,
                use_gpu_embeddings=False  # embeddings on CPU
            )

    # ——— 2️⃣ Generate response ———
    try:
        resp = chatbot.generate_response(
            question, engine,
            data.get("conversation_history", []),
            data.get("target_title")
        )
        answer = resp.get("chatbot_response", "")
        return jsonify({"answer": answer})
    finally:
        # ——— 3️⃣ Reschedule unload for after IDLE_UNLOAD_DELAY ———
        _schedule_unload()



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
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    hashed_password = pbkdf2_sha256.hash(password)

    try:
        with engine.begin() as conn:
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
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
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
        # only consider audio files! This is done to skip over the audio_status_files
        if filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            file_url = request.host_url + f"audio/{user_folder}/{filename}"
            audios.append({"filename": filename, "url": file_url})
    return jsonify({"audios": audios}), 200

# ------------------------------
# Delete Audio
# ------------------------------

@app.route('/api/delete-audio', methods=['POST'])
def delete_audio():
    data = request.get_json()
    username = data.get("username")
    filename = data.get("filename")
    if not username or not filename:
        return jsonify({"error": "Username and filename required"}), 400

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

# ------------------------------
# Process Audio Endpoint
# ------------------------------

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """
    Triggers the audio processing pipeline for a target audio.
    Expects JSON with 'username' and 'filename'.
    Uses a Windows batch file (process_audio.bat) to run the pipeline.
    Runs the process in a background thread so that Flask returns immediately.
    The title is derived from the filename (without extension) to reference the clip in the DB.
    """
    data = request.get_json()
    username = data.get("username")
    filename = data.get("filename")
    if not username or not filename:
        return jsonify({"error": "Username and filename required"}), 400

    with engine.connect() as conn:
        result = conn.execute(text("SELECT storage_folder FROM users WHERE username = :username"), {"username": username})
        row = result.fetchone()
        if not row:
            return jsonify({"error": "User not found"}), 404
        user_folder = row[0]

    audio_path = os.path.join(UPLOAD_FOLDER, user_folder, filename)
    if not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found"}), 404

    # Get the title from the filename (without extension)
    title, _ = os.path.splitext(filename)

    # Define a background function to run the pipeline.
    def run_pipeline():
        current_os = platform.system()
        try:
            print(f"Detected platform OS: {current_os}")
            user_dir = os.path.join(UPLOAD_FOLDER, user_folder)
            if current_os == "Windows":
                # on Windows, use the batch script from the project root
                script_path = os.path.join(os.getcwd(), "process_audio.bat")
                subprocess.run(
                    ["cmd", "/c", script_path, audio_path, username, title],
                    cwd=user_dir,
                    check=True
                )
            else:
                script_path = os.path.join(os.getcwd(), "linux_process_audio.sh")
                # run from inside the user’s folder so that status_<title>.txt ends up there
                subprocess.run(
                  [script_path, audio_path, username, title],
                  cwd=user_dir, check=True
                )
            print("Audio processing completed successfully for:", title)

        except subprocess.CalledProcessError as e:
            print(f"Error: Script execution failed with exit code {e.returncode}")
        except Exception as e:
            print("Error processing audio in background thread:", str(e))

    # Start the pipeline in a background thread.
    import threading
    threading.Thread(target=run_pipeline).start()

    return jsonify({"success": True, "message": "Audio processing started. It may take a few minutes."}), 200


@app.route('/api/audio-status', methods=['POST'])
def audio_status():
    """
Retrieve the current processing status of a user's audio file.

Expects a JSON payload containing:
    - username (str): the user's username.
    - filename (str): the exact name of the uploaded audio file (including extension).

Workflow:
1. Validate that both 'username' and 'filename' are provided.
    - If missing, returns HTTP 400 with {"error": "..."}.
2. Look up the user's storage folder in the database.
    - If the user is not found, returns HTTP 404 with {"error": "User not found"}.
3. Construct the path to the status file: 
        <UPLOAD_FOLDER>/<storage_folder>/status_<title>.txt
    where <title> is the filename without its extension.
4. If the status file exists:
        - Read its contents, strip whitespace, and return it as {"status": <content>}.
    Otherwise:
        - Return {"status": "Not started"}.
5. Always returns HTTP 200 on success.

Returns:
    JSON response with a single 'status' field indicating the pipeline step,
    or an appropriate error message and HTTP status code on failure.
"""
    data = request.get_json()
    username = data.get("username")
    filename = data.get("filename")
    if not username or not filename:
        return jsonify({"error": "username and filename required"}), 400

    # look up the user’s folder
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT storage_folder FROM users WHERE username=:username"),
            {"username": username}
        )
        row = result.fetchone()
    if not row:
        return jsonify({"error": "User not found"}), 404

    user_folder = row[0]
    title, _ = os.path.splitext(filename)
    status_path = os.path.join(UPLOAD_FOLDER, user_folder, f"status_{title}.txt")

    if os.path.exists(status_path):
        with open(status_path) as f:
            status = f.read().strip()
    else:
        # ---------- NEW: fall‑back to database ----------
        #   An audio is considered "processed" when
        #   1) originalAudio contains a row whose title matches <title>
        #   2) at least one clip exists that references that audio_id.
        with engine.connect() as conn:
            query = text("""
                SELECT COUNT(*)   -- clips_cnt
                FROM originalaudio  o
                JOIN clips c USING (audio_id)
                WHERE o.title = :title
            """)
            # in the line below, if .scalar() returns a Falsey value, it will be replaced by 0 (because of 'or 0')
            #   (e.g. None or empty string).
            clips_cnt = conn.execute(query, {"title": title}).scalar() or 0
        status = "Completed" if clips_cnt > 0 else "Not started"

    return jsonify({"status": status}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
