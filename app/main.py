import streamlit as st
import requests
import os
import time

# Configuration – point to your Flask backend.
BACKEND_URL = "http://localhost:5000"

st.image("app/assets/logo.png", width=750)  # Adjust the width as needed

# ------------------------------
# Login/Registration Section
# ------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("Login / Registration")
    auth_mode = st.radio("Select mode:", ("Login", "Register"))
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if auth_mode == "Login":
        if st.button("Login"):
            response = requests.post(f"{BACKEND_URL}/api/login", json={"username": username, "password": password})
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    st.success("Logged in successfully!")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.chat_history = []
                    st.session_state.target_title = None
                    st.rerun()
                else:
                    st.error(data.get("message", "Login failed"))
            else:
                st.error("Login request failed.")
    else:  # Registration mode
        if st.button("Register"):
            response = requests.post(f"{BACKEND_URL}/api/register", json={"username": username, "password": password})
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    st.success("Registration successful! Please log in.")
                else:
                    st.error(data.get("message", "Registration failed"))
            else:
                st.error("Registration request failed.")
    st.stop()

# ------------------------------
# Main Application (After Login)
# ------------------------------

# Sidebar: Audio Upload, Listing, Deletion, and Processing
with st.sidebar:
    st.header("Audio Features")
    
    # --- Audio Upload Section ---
    st.subheader("Upload Audio")
    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        if st.button("Upload & Send to Server"):
            files = {"file": (audio_file.name, audio_file.read())}
            data = {"username": st.session_state.username}
            response = requests.post(f"{BACKEND_URL}/upload-audio", data=data, files=files)
            if response.status_code == 200:
                res_data = response.json()
                if "url" in res_data:
                    st.success("File uploaded successfully!")
                    # immediately start processing
                    proc = requests.post(
                        f"{BACKEND_URL}/api/process-audio",
                        json={
                          "username": st.session_state.username,
                          "filename": audio_file.name
                        }
                    )
                    if proc.status_code == 200:
                        st.info("Audio processing started…")
                       # remember which file is in flight
                        st.session_state.processing_audio = audio_file.name
                    else:
                        st.error("Failed to start processing.")
                    st.rerun()
            else:
                st.error("Upload failed. Please check the server.")

    # --- List User Audios ---
    st.subheader("My Uploaded Audios")
    list_response = requests.post(
        f"{BACKEND_URL}/api/list-audios",
        json={"username": st.session_state.username}
    )
    if list_response.status_code == 200:
        audios = list_response.json().get("audios", [])

        if not audios:
            st.info("No uploaded audios yet.")
        else:
            for audio in audios:
                fname = audio["filename"]
                st.markdown(f"**{fname}**")
                st.audio(audio["url"])

                # 1) If this is the file currently uploading/processing, show its live status
                if st.session_state.get("processing_audio") == fname:
                    status_res = requests.post(
                        f"{BACKEND_URL}/api/audio-status",
                        json={
                            "username": st.session_state.username,
                            "filename": fname
                        }
                    )
                    status = status_res.json().get("status", "…")
                    st.warning(f"🔄 Processing step: **{status}**")

                    if status == "Completed":
                        st.success(f"✅ {fname} has been processed!")
                        # clear the flag so the chat box and Select‑as‑Target re‑appear
                        del st.session_state["processing_audio"]
                        st.rerun()
                    else:
                        # audio is STILL processing
                        time.sleep(2) # pause before re-polling
                        st.rerun()

                else:
                    # 2) For already‑uploaded files, check if they're done
                    status_res = requests.post(
                        f"{BACKEND_URL}/api/audio-status",
                        json={
                            "username": st.session_state.username,
                            "filename": fname
                        }
                    )
                    if status_res.json().get("status") == "Completed":
                        st.info("Ready for Q&A")
                        # allow them to select it now
                        if st.button("Select as Target", key=f"select_{fname}"):
                            st.session_state.target_title = os.path.splitext(fname)[0]
                            st.success(f"Target audio set to: {st.session_state.target_title}")
                            st.rerun()
                    else:
                        # status_{title}.txt does not exist in the user's uploaded_audios folder, assume we have not started processing yet.
                        st.info("Not yet processed")

                # 3) Always let them delete
                if st.button("Delete", key=f"delete_{fname}"):
                    del_response = requests.post(
                        f"{BACKEND_URL}/api/delete-audio",
                        json={
                            "username": st.session_state.username,
                            "filename": fname
                        }
                    )
                    if del_response.status_code == 200:
                        st.success("File deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete file.")

    else:
        st.error("Failed to list audios.")


    # Optionally display currently selected target
    if st.session_state.get("target_title"):
        st.info(f"Current target audio for Q&A: {st.session_state.target_title}")

# Main page: Chatbot Interface
st.header("Conversational Insights from Recorded Lectures")
st.markdown(
    "<h4 style='font-weight:normal; font-size:20px;'>Upload your audio file and process it to select it as target</h4>", 
    unsafe_allow_html=True
)



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation history using keys "user" and "chatbot_response"
for exchange in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(exchange["user"])
    with st.chat_message("assistant"):
        st.markdown(exchange["chatbot_response"])

# … after you re‐render the conversation history …

# only allow questions once processing is done
if "processing_audio" in st.session_state:
    st.info("Please wait until your audio file has finished processing.")
else:
    # Chat input field using st.chat_input
    chat_input = st.chat_input("Enter your question:")

    if chat_input:
        st.session_state.chat_history.append({"user": chat_input, "chatbot_response": ""})
        payload = {
            "question": chat_input,
            "conversation_history": st.session_state.chat_history,
            "target_title": st.session_state.get("target_title")
        }
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer returned.")
            st.session_state.chat_history[-1]["chatbot_response"] = answer
            st.rerun()
        else:
            st.error("Error getting chatbot response.")


#  Error: Input file  does not exist
# 20:06:45 backend.1  | Error: Input file output_files/transcript_GeminiTalk_espa.json does not exist
# 20:06:47 backend.1  | Error: Input file output_files/chunks_GeminiTalk_espa.csv does not exist
# 20:06:51 backend.1  | Error: File output_files/original_audio_GeminiTalk_espa.csv does not exist