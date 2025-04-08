# --- START OF MERGED chatbot.py ---

# Imports from both Code 1 and Code 2
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings # Included from Code 1, although commented out in Code 2's version
from sentence_transformers import SentenceTransformer
import psycopg2
import re
import argparse
from sqlalchemy import create_engine
from docx import Document # Added for .docx output (from Code 2)
from docx.shared import Pt # Added for font size if needed (optional, from Code 2)
from tqdm import tqdm # Added for progress bar during testing (from Code 2)

class Chatbot:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", llm_model="deepseek-r1:7b", llm_base_url="http://localhost:11434"):
        """
        Initialize the Chatbot with an embedding model and an LLM.
        (Includes initialization prints from Code 2)
        """
        print(f"Initializing Chatbot with embedding model: {embedding_model_name} and LLM: {llm_model}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm = ChatOllama(model=self.llm_model, temperature=0.5, base_url=self.llm_base_url)
        print("Chatbot initialized.")

    def get_fragments_from_question(self, question, engine, target_title=None):
        """
        Extract fragments from the PostgreSQL table 'Clips' based on the embedding similarity.
        If target_title is provided (from Code 1), only clips with that title are retrieved.
        Uses LIMIT 3 (from Code 2) and error handling (from Code 2).
        """
        # print(f"Generating embedding for question: '{question[:50]}...'") # Optional print from Code 2
        emb = self.embedding_model.encode(
            question,
            # show_progress_bar=True, # Can be noisy in loops, disabled for testing loop (from Code 2)
            batch_size=32,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        # Convert the NumPy array to a string acceptable by pgvector
        embedding_str = "[" + ",".join(map(str, emb)) + "]"
        conn_str = engine.url.render_as_string(hide_password=False)
        conn = None
        result = []
        try:
            # print("Connecting to database...") # Optional print from Code 2
            conn = psycopg2.connect(conn_str)
            cur = conn.cursor()

            if target_title:
                # Query logic from Code 1 (with LIMIT 3 from Code 2)
                query = """
                SELECT c.clip_id, c.transcript,
                    1 - (c.embedding <-> %s) AS similarity
                FROM Clips c
                JOIN originalAudio o ON c.audio_id = o.audio_id
                WHERE o.title = %s
                ORDER BY c.embedding <-> %s
                LIMIT 3
                """
                # print("Executing similarity search query with title filter...") # Modified optional print
                cur.execute(query, (embedding_str, target_title, embedding_str))
            else:
                # Query logic from Code 2 (modified from Code 1's else branch)
                query = """
                SELECT clip_id, transcript,
                       1 - (embedding <-> %s) AS similarity
                FROM Clips
                ORDER BY embedding <-> %s
                LIMIT 3
                """
                # print("Executing similarity search query without title filter...") # Modified optional print
                cur.execute(query, (embedding_str, embedding_str))

            result = cur.fetchall()
            # print(f"Retrieved {len(result)} fragments.") # Optional print from Code 2
            cur.close()
        except psycopg2.Error as e:
            # Error handling from Code 2
            print(f"Database error during fragment retrieval: {e}")
            # Depending on requirements, you might want to raise the error or return empty
        finally:
            # Connection closing from Code 2
            if conn:
                conn.close()
                # print("Database connection closed.") # Optional print from Code 2
        return result

    def generate_response(self, user_question, engine, conversation_history=[], target_title=None):
        """
        Generate a response by retrieving context from clips and feeding it into an LLM.
        Includes conversation history (from both).
        Allows filtering by target_title (from Code 1).
        Uses context separator '---' (from Code 2).
        Uses prompt template from Code 2.
        Returns context_used (from Code 2).
        Includes LLM invocation error handling (from Code 2).
        Uses history character limit of 2000 (consistent in both effective implementations).
        """
        # Retrieve context fragments (filtered by target_title if provided - merging Code 1 logic)
        fragments = self.get_fragments_from_question(user_question, engine, target_title)
        context_list = [fragment[1] for fragment in fragments]
        # Use clear separator for context pieces (from Code 2)
        documents = "\n\n---\n\n".join(context_list)

        # --- CONTEXT CAPTURE FOR TESTING (from Code 2) ---
        # Store the raw context string to return it alongside the response
        context_used_for_response = documents if documents else "No context retrieved."
        # --- END CONTEXT CAPTURE ---

        # print(f"Context retrieved for LLM: {documents[:200]}...") # Optional print from Code 2

        # Build conversation history string up to 2000 characters.
        history_entries = []
        total_length = 0
        for entry in reversed(conversation_history):
            # Ensure keys exist, provide defaults if not (robustness from Code 2)
            user_msg = entry.get('user', '[User message missing]')
            bot_msg = entry.get('chatbot_response', '[Bot response missing]')
            msg = f"User: {user_msg}\nAssistant: {bot_msg}\n{'-'*35}\n"
            if total_length + len(msg) > 2000: # Limit history size
                break
            history_entries.append(msg)
            total_length += len(msg)
        # Reverse back so that messages appear in chronological order
        conversation_history_str = "".join(reversed(history_entries)) if history_entries else "None"

        # Define the prompt including conversation history (using Code 2's template)
        prompt_template = PromptTemplate(
            template="""You are an AI assistant specialized in answering questions based *only* on the provided context. 
            The provided context is from a transcript of an audio file. If the context does not contain the answer, 
            say you don't have enough information in the provided text. You also have a conversation history for context. 
            Answer in the language you are spoken to. 
Conversation History:
{conversation_history}

Provided Context:
{documents}

Question: {user_question}
Answer:""",
            input_variables=["user_question", "documents", "conversation_history"]
        )
        rag_chain = prompt_template | self.llm | StrOutputParser()

        # print("Invoking RAG chain...") # Optional print from Code 2
        try:
            answer = rag_chain.invoke({
                "user_question": user_question,
                "documents": documents,
                "conversation_history": conversation_history_str,
            })
            # Basic cleaning (remove potential XML-like thinking tags if LLM adds them)
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            # print("RAG chain invocation complete.") # Optional print from Code 2
        except Exception as e:
            # LLM error handling from Code 2
            print(f"Error invoking LLM chain: {e}")
            answer = "An error occurred while generating the response."

        # Return the standard response structure PLUS the context used (from Code 2)
        return {
            "chatbot_response": answer,
            "context_used": context_used_for_response # Added for testing (from Code 2)
            }

# --- TESTING BLOCK (from Code 2) ---
def run_predefined_tests(chatbot_instance, engine, questions, responses_filename="chatbot_responses.docx", context_filename="context_per_question.docx"):
    """
    Runs a list of predefined questions through the chatbot and saves
    responses and context used into separate .docx files.
    (Function entirely from Code 2)
    """
    print(f"\n--- Starting Predefined Tests ({len(questions)} questions) ---")

    # Initialize .docx documents
    responses_doc = Document()
    context_doc = Document()

    responses_doc.add_heading('Chatbot Test Responses', level=1)
    context_doc.add_heading('Context Used Per Question', level=1)

    for i, question in enumerate(tqdm(questions, desc="Processing Questions")):
        print(f"\nProcessing Question {i+1}/{len(questions)}: {question}")

        # Generate response using empty conversation history
        # NOTE: target_title is not used in test mode based on Code 2's original test setup.
        # If title-specific testing is needed, this call would need modification.
        try:
            response_data = chatbot_instance.generate_response(question, engine, conversation_history=[]) # No target_title here
            chatbot_response = response_data.get("chatbot_response", "Error: Response key not found.")
            context_used = response_data.get("context_used", "Error: Context key not found.")
        except Exception as e:
             print(f"  [ERROR] Failed to get response for question {i+1}: {e}")
             chatbot_response = f"Error during generation: {e}"
             context_used = "N/A due to error"

        # --- Add to Responses Doc ---
        p_resp_q = responses_doc.add_paragraph()
        p_resp_q.add_run(f"{i+1}. ").bold = True
        p_resp_q.add_run(question).bold = True
        responses_doc.add_paragraph(chatbot_response)
        responses_doc.add_paragraph() # Add spacing

        # --- Add to Context Doc ---
        p_cont_q = context_doc.add_paragraph()
        p_cont_q.add_run(f"{i+1}. ").bold = True
        p_cont_q.add_run(question).bold = True
        context_doc.add_paragraph(context_used)
        context_doc.add_paragraph() # Add spacing

    # Save documents
    try:
        responses_doc.save(responses_filename)
        print(f"\nChatbot responses saved to: {responses_filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save responses document '{responses_filename}': {e}")

    try:
        context_doc.save(context_filename)
        print(f"Context used saved to: {context_filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save context document '{context_filename}': {e}")

    print("--- Predefined Tests Finished ---")


if __name__ == "__main__":
    # Argparse setup from Code 2 (more comprehensive)
    parser = argparse.ArgumentParser(description="Chatbot - Run in interactive mode or execute predefined tests.")
    parser.add_argument("--db_uri", required=True, help="PostgreSQL connection URI (e.g., postgresql://user:pass@host/dbname)")
    parser.add_argument("--mode", choices=['interactive', 'test'], default='interactive', help="Run mode: 'interactive' for live chat, 'test' to run predefined questions.")
    parser.add_argument("--llm", default="deepseek-r1:7b", help="LLM model name (e.g., deepseek-r1:7b, deepseek-r1:14b)") # Allow choosing LLM (from Code 2)
    parser.add_argument("--title", default=None, help="Optional: Filter context by audio title in interactive mode.") # Added to allow using target_title in interactive mode

    args = parser.parse_args()

    print("Creating database engine...")
    try:
        # Engine creation and connection test from Code 2
        engine = create_engine(args.db_uri)
        with engine.connect() as connection:
             print("Database connection successful.")
    except Exception as e:
        print(f"Failed to create database engine or connect: {e}")
        exit(1) # Exit if DB connection fails

    print("Initializing chatbot...")
    # Pass the selected LLM model to the chatbot constructor (from Code 2)
    chatbot = Chatbot(llm_model=args.llm)

    if args.mode == 'test':
        # Test mode execution from Code 2
        # --- List of Predefined Questions (using the second list defined in Code 2) ---
        # Note: Code 2 had two lists, the second overwrites the first. Using the second one.
        predefined_questions = [
            "What personal experience sparked the speaker's interest in addiction?",
            "Why did the speaker travel around the world, and who did he meet?",
            "What is the traditional view of addiction that the speaker challenges?",
            "How does the example of diamorphine (medical heroin) contradict the traditional view of addiction?",
            "What was the Rat Park experiment, and what did it show?",
            "What human example supports the findings of the Rat Park experiment?",
            "What alternative explanation for addiction does the speaker propose?",
            "What was Portugal's approach to dealing with addiction, and what were the results?",
            "How does our culture typically respond to addiction, according to the speaker?",
            "What is the speaker's main message about how we should treat addicts?",
            "Why does the speaker criticize the show 'Intervention'?",
            "What does the speaker suggest is the true opposite of addiction?",
            "How does the speaker relate modern society to the Rat Park experiment?",
            "What societal trends does the speaker mention as contributing to disconnection?",
            "What did the speaker learn about helping loved ones with addiction?"
        ]
        # --- End of Predefined Questions ---

        # Run the tests
        run_predefined_tests(chatbot, engine, predefined_questions)

    elif args.mode == 'interactive':
         # Interactive mode execution based on Code 2 (maintains history)
         print("\n--- Starting Interactive Mode (Ctrl+C to exit) ---")
         if args.title:
             print(f"--- Filtering context by title: {args.title} ---")
         conversation_history = [] # Keep track of history in interactive mode
         while True:
            try:
                question = input("\nAsk the chatbot a question: ")
                if question.lower() in ['exit', 'quit']:
                     break
                # Pass current history and get response
                # Pass target_title if provided via command line argument
                response_data = chatbot.generate_response(
                    question,
                    engine,
                    conversation_history=conversation_history,
                    target_title=args.title # Pass title filter if specified
                )
                chatbot_response = response_data.get("chatbot_response", "Sorry, I encountered an issue.")

                print("\nRESPONSE:")
                print(chatbot_response)

                # Add interaction to history for the next turn
                conversation_history.append({
                     "user": question,
                     "chatbot_response": chatbot_response
                })
                # Optional: Prune history if it gets too long over many turns (from Code 2 comment)
                # MAX_HISTORY = 10 # Keep last 10 interactions
                # if len(conversation_history) > MAX_HISTORY:
                #      conversation_history = conversation_history[-MAX_HISTORY:]

            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                 print(f"\nAn error occurred: {e}")
                 # Optionally reset history or handle error more gracefully

# Example command lines from Code 2 (updated to show --title usage)
# python chatbot.py --db_uri postgresql://admin:secret@localhost:5432/testdb --mode test --llm "deepseek-r1:14b"
# python chatbot.py --db_uri postgresql://admin:secret@localhost:5432/testdb --mode interactive --llm "deepseek-r1:14b"
# python chatbot.py --db_uri postgresql://admin:secret@localhost:5432/testdb --mode interactive --llm "deepseek-r1:7b" --title "My Specific Lecture Title"

# --- END OF MERGED chatbot.py ---