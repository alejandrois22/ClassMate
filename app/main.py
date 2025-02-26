import streamlit as st
import requests

# Configuration
BACKEND_URL = "http://localhost:8000"        
SERVER_AUDIO_URL = "http://localhost:8080/audio/"

# Set up the page title logo/image
st.title("ClassMate - Conversational Insights from Recorded Lectures")
# st.image("logo.png", width=200)  #logo

# --- Audio Upload Section ---
st.header("Upload Audio")
audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    # Preview the audio locally
    st.audio(audio_file, format="audio/wav")
    
    if st.button("Upload & Send to Server"):
        # Read file content
        files = {"file": (audio_file.name, audio_file.read())}
        response = requests.post(f"{BACKEND_URL}/upload-audio/", files=files)
        
        if response.status_code == 200:
            data = response.json()
            # st.write("Server response:", data)  # Debugging
            if "url" in data:
                audio_url = data["url"].replace("http//", "http://").replace("//audio/", "/audio/")
                st.success("File uploaded successfully!")
                # Embed an audio player that plays from the server URL
                st.audio(audio_url)
            else:
                st.error("Response does not contain 'url'. Check the server response.")
                st.write("Full response:", data)
        else:
            st.error("Upload failed. Please check the server.")

# --- Audio Retrieval Section ---
st.header("Retrieve Stored Audio")
filename = st.text_input("Enter audio filename (e.g., lecture1.opus)")
if st.button("Retrieve"):
    response = requests.get(f"{BACKEND_URL}/get-audio/{filename}")
    if response.status_code == 200:
        data = response.json()
        if "url" in data:
            # Embed an audio player for the retrieved file
            st.audio(data["url"])
        else:
            st.error("Response does not contain 'url'.")
    else:
        st.error("File not found.")

# --- Chatbot Interface ---
st.header("Chatbot Interface")

# Initialize chat history in session state if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.subheader("Conversation History")
for chat in st.session_state.chat_history:
    st.markdown(f"**{chat['sender']}**: {chat['message']}")

# Input for new chat message
chat_input = st.text_input("Enter your message", key="chat_input")
if st.button("Send"):
    if chat_input:
        # Append the user's message
        st.session_state.chat_history.append({"sender": "User", "message": chat_input})
        # Placeholder response for the chatbot (replace with actual logic later)
        bot_response = "This is a placeholder response."
        st.session_state.chat_history.append({"sender": "Bot", "message": bot_response})
        # Clear the input field
        st.session_state.chat_input = ""
        # Rerun to update display (optional, depending on your design)
        st.experimental_rerun()
