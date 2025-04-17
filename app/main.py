import streamlit as st
import requests
import os
import math
import io # for downloading the conversation as txt
from st_audiorec import st_audiorec
import speech_recognition as sr
import tempfile

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

    st.success(f"Welcome, {st.session_state.username}! 👋", icon="✅")
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
                    st.rerun()
                else:
                    st.error("Response does not contain 'url'.")
            else:
                st.error("Upload failed. Please check the server.")

    # --- List User Audios ---
    st.subheader("My Uploaded Audios")
    list_response = requests.post(f"{BACKEND_URL}/api/list-audios", json={"username": st.session_state.username})
    if list_response.status_code == 200:
        audios = list_response.json().get("audios", [])
        if audios:
            # Create a list of target options (using filename without extension)
            target_options = {}
            for audio in audios:
                st.markdown(f"**{audio['filename']}**")
                st.audio(audio["url"])
                st.caption(f"Filename: {audio['filename']}")
                size_mb = audio.get("size", 0) / 1_000_000 if "size" in audio else None
                if size_mb:
                    st.caption(f"Size: {math.ceil(size_mb*100)/100} MB")
                # Button to process audio (runs pipeline)
                if st.button("Process Audio", key=f"process_{audio['filename']}"):
                    process_response = requests.post(
                        f"{BACKEND_URL}/api/process-audio",
                        json={"username": st.session_state.username, "filename": audio["filename"]}
                    )
                    if process_response.status_code == 200:
                        st.success(process_response.json().get("message", "Audio processed."))
                        st.rerun()
                    else:
                        st.error("Failed to process audio.")
                # Button to select as target
                if st.button("Select as Target", key=f"select_{audio['filename']}"):
                    title, _ = os.path.splitext(audio["filename"])
                    st.session_state.target_title = title
                    st.success(f"Target audio set to: {title}")
                    st.rerun()
                # Add to options list for later use if desired
                target_options[audio["filename"]] = os.path.splitext(audio["filename"])[0]
                # Button to delete audio
                if st.button("Delete", key=f"delete_{audio['filename']}"):
                    del_response = requests.post(
                        f"{BACKEND_URL}/api/delete-audio",
                        json={"username": st.session_state.username, "filename": audio["filename"]}
                    )
                    if del_response.status_code == 200:
                        st.success("File deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete file.")
        else:
            st.info("No uploaded audios yet.")
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
        st.markdown(f"**🧑 You:** {exchange['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**🤖 ClassMate:** {exchange['chatbot_response']}")
    
# Clear Chat Button
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
    st.rerun()




# Chat input field using st.chat_input
chat_input = st.chat_input("Enter your question:")
if chat_input:
    # Add user message immediately to chat
    st.session_state.chat_history.append({
        "user": chat_input,
        "chatbot_response": "..."  # Placeholder while waiting for response
    })
    st.rerun()  # Triggers page refresh so it shows user's message right away

# Handle incomplete chatbot responses
for exchange in st.session_state.chat_history:
    if exchange["chatbot_response"] == "...":
        payload = {
            "question": exchange["user"],
            "conversation_history": st.session_state.chat_history,
            "target_title": st.session_state.get("target_title")
        }
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            exchange["chatbot_response"] = data.get("answer", "No answer returned.")
            st.rerun()
        else:
            exchange["chatbot_response"] = "❌ Error getting chatbot response."
            st.rerun()
        break  # Only process one at a time

if st.download_button("📄 Download Chat", 
                      data="\n\n".join([f"You: {x['user']}\nClassMate: {x['chatbot_response']}" 
                                        for x in st.session_state.chat_history]),
                      file_name="chat_history.txt"):
    st.success("Download started!")