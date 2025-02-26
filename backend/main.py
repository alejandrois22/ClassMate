from fastapi import FastAPI, File, UploadFile
import requests

app = FastAPI()


UPLOAD_URL = "http://localhost:8080/upload"
# This URL is used to access the stored audio files
AUDIO_ACCESS_URL = "http://localhost:8080/audio/"

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """ Sends audio file from FastAPI (Windows) to the Debian server. """
    files = {"file": (file.filename, file.file.read())}
    response = requests.post(UPLOAD_URL, files=files)
    
    # Debugging: print the Debian server response
    print("Debian Server Response:", response.status_code, response.text)
    
    if response.status_code == 200:
        # Return the response from the server (which includes the file URL)
        return response.json()
    return {
        "error": "Upload failed",
        "status_code": response.status_code,
        "response": response.text
    }

@app.get("/get-audio/{filename}")
async def get_audio(filename: str):
    """ Returns the URL to access the stored audio file on Debian. """
    correct_url = f"{AUDIO_ACCESS_URL}{filename}"
    return {"url": correct_url}
