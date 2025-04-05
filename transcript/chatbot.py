# --- START OF FILE chatbot.py ---

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import OllamaEmbeddings # Not directly used in this version
from sentence_transformers import SentenceTransformer
import psycopg2
import re
import argparse
from sqlalchemy import create_engine
from docx import Document # Added for .docx output
from docx.shared import Pt # Added for font size if needed (optional)
from tqdm import tqdm # Added for progress bar during testing

class Chatbot:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", llm_model="deepseek-r1:7b", llm_base_url="http://localhost:11434"):
        """
        Initialize the Chatbot with an embedding model and an LLM.
        """
        print(f"Initializing Chatbot with embedding model: {embedding_model_name} and LLM: {llm_model}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm = ChatOllama(model=self.llm_model, temperature=0.5, base_url=self.llm_base_url)
        print("Chatbot initialized.")

    def get_fragments_from_question(self, question, engine):
        """
        Extract fragments from the PostgreSQL table 'Clips' based on the embedding similarity.
        """
        # print(f"Generating embedding for question: '{question[:50]}...'")
        emb = self.embedding_model.encode(
            question,
            # show_progress_bar=True, # Can be noisy in loops, disabled for testing loop
            batch_size=32,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        # Convert the NumPy array to a string acceptable by pgvector
        embedding_str = "[" + ",".join(map(str, emb)) + "]"
        conn_str = engine.url.render_as_string(hide_password=False)
        conn = None
        result = []
        try:
            # print("Connecting to database...")
            conn = psycopg2.connect(conn_str)
            cur = conn.cursor()
            query = """
            SELECT clip_id, transcript,
                   1 - (embedding <-> %s) AS similarity
            FROM Clips
            ORDER BY embedding <-> %s
            LIMIT 3
            """
            # print("Executing similarity search query...")
            cur.execute(query, (embedding_str, embedding_str))
            result = cur.fetchall()
            # print(f"Retrieved {len(result)} fragments.")
            cur.close()
        except psycopg2.Error as e:
            print(f"Database error during fragment retrieval: {e}")
            # Depending on requirements, you might want to raise the error or return empty
        finally:
            if conn:
                conn.close()
                # print("Database connection closed.")
        return result

    def generate_response(self, user_question, engine, conversation_history=[]):
        """
        Generate a response by retrieving context from clips and feeding it into an LLM.

        This version adds conversation history into the prompt.
        It accumulates the most recent messages (starting from the newest)
        until a 2,000-character context window is reached, then displays them in chronological order.

        MODIFIED FOR TESTING: Also returns the context used under the key 'context_used'.
                              The primary 'chatbot_response' key remains for compatibility.
        """
        # Retrieve context fragments
        fragments = self.get_fragments_from_question(user_question, engine)
        context_list = [fragment[1] for fragment in fragments]
        documents = "\n\n---\n\n".join(context_list) # Use a clear separator for context pieces

        # --- CONTEXT CAPTURE FOR TESTING ---
        # Store the raw context string to return it alongside the response
        context_used_for_response = documents if documents else "No context retrieved."
        # --- END CONTEXT CAPTURE ---

        # print(f"Context retrieved for LLM: {documents[:200]}...") # Print start of context if needed

        # Build conversation history string up to 20000 characters (NOTE: Original code had 20000, might be too large, check LLM limits)
        # Using 2000 as mentioned in original description seems more reasonable. Let's stick to 2000.
        history_entries = []
        total_length = 0
        for entry in reversed(conversation_history):
            # Ensure keys exist, provide defaults if not
            user_msg = entry.get('user', '[User message missing]')
            bot_msg = entry.get('chatbot_response', '[Bot response missing]')
            msg = f"User: {user_msg}\nAssistant: {bot_msg}\n{'-'*35}\n"
            if total_length + len(msg) > 2000: # Limit history size
                break
            history_entries.append(msg)
            total_length += len(msg)
        # Reverse back so that messages appear in chronological order
        conversation_history_str = "".join(reversed(history_entries)) if history_entries else "None"

        # Define the prompt including conversation history.
        prompt_template = PromptTemplate(
            template="""You are an AI assistant specialized in answering questions based *only* on the provided context. If the context does not contain the answer, say you don't have enough information in the provided text. You also have a conversation history for context.
Conversation History:
{conversation_history}

Provided Context:
{documents}

Question: {user_question}
Answer:""",
            input_variables=["user_question", "documents", "conversation_history"]
        )
        rag_chain = prompt_template | self.llm | StrOutputParser()

        # print("Invoking RAG chain...")
        try:
            answer = rag_chain.invoke({
                "user_question": user_question,
                "documents": documents,
                "conversation_history": conversation_history_str,
            })
            # Basic cleaning (remove potential XML-like thinking tags if LLM adds them)
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            # print("RAG chain invocation complete.")
        except Exception as e:
            print(f"Error invoking LLM chain: {e}")
            answer = "An error occurred while generating the response."

        # Return the standard response structure PLUS the context used
        return {
            "chatbot_response": answer,
            "context_used": context_used_for_response # Added for testing
            }

# --- TESTING BLOCK ---
def run_predefined_tests(chatbot_instance, engine, questions, responses_filename="chatbot_responses.docx", context_filename="context_per_question.docx"):
    """
    Runs a list of predefined questions through the chatbot and saves
    responses and context used into separate .docx files.
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
        try:
            response_data = chatbot_instance.generate_response(question, engine, conversation_history=[])
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
    parser = argparse.ArgumentParser(description="Chatbot - Run in interactive mode or execute predefined tests.")
    parser.add_argument("--db_uri", required=True, help="PostgreSQL connection URI (e.g., postgresql://user:pass@host/dbname)")
    parser.add_argument("--mode", choices=['interactive', 'test'], default='interactive', help="Run mode: 'interactive' for live chat, 'test' to run predefined questions.")
    parser.add_argument("--llm", default="deepseek-r1:7b", help="LLM model name (e.g., deepseek-r1:7b, deepseek-r1:14b)") # Allow choosing LLM
    args = parser.parse_args()

    print("Creating database engine...")
    try:
        engine = create_engine(args.db_uri)
        # Optional: Test connection early
        with engine.connect() as connection:
             print("Database connection successful.")
    except Exception as e:
        print(f"Failed to create database engine or connect: {e}")
        exit(1) # Exit if DB connection fails

    print("Initializing chatbot...")
    # Pass the selected LLM model to the chatbot constructor
    chatbot = Chatbot(llm_model=args.llm)

    if args.mode == 'test':
        # --- List of Predefined Questions ---
        predefined_questions = [
            "What is the proposed tariff on goods from China?",
            "What is the proposed tariff on goods from Mexico?",
            "What is the proposed tariff on goods from Canada?",
            "What is the proposed tariff on steel and aluminum?",
            "Who pays tariffs on imported goods?",
            "What was the Smoot-Hawley Tariff Act?",
            "Did the Smoot-Hawley Tariff Act work?",
            "What was a major economic consequence of Trump’s steel tariffs?",
            "How much did U.S. steel production grow annually due to Trump’s steel tariffs?",
            "What was the estimated real household income loss in 2021 due to Trump’s tariffs?",
            "How much revenue did the U.S. government collect from tariffs on China in Obama’s last year?",
            "What is the U.S. trade deficit?",
            "What country has the largest trade deficit with the U.S.?",
            "What industries in the U.S. were impacted by increased steel prices?",
            "How much could Trump's various tariff plans cost the average American household per year?",
            "What is one reason U.S. manufacturing jobs declined over the last 50 years?",
            "What is one economic risk of widespread tariffs?",
            "What do proponents of tariffs cite as a successful historical period for tariffs?",
            "What is one argument against the historical success of tariffs?",
            "What are some unfair trade practices China has been accused of?",
            "What did some American farmers receive during Trump’s first term due to tariffs?",
            "What is a potential political issue with tariff exemptions?",
            "How did some companies gain an advantage under Trump’s tariff policy?",
            "What is one possible alternative use of tariff revenue mentioned in the video?",
            "What is one recommendation given to consumers to prepare for tariffs?",
            "How did Canada and Mexico respond to Trump’s tariffs in early February?",
            "What is one concern about the effectiveness of tariffs as an economic tool?",
            "What is a potential long-term consequence of tariffs on U.S. manufacturing?",
            "What is a major labor-related issue that could hinder U.S. manufacturing growth?",
            "How did Trump justify his tariffs besides economic reasons?",
            "What type of businesses are likely to pass tariff costs to consumers?",
            "How did retaliatory tariffs affect U.S. exports?",
            "How did the late 19th-century tariff policies lead to corruption?",
            "How did COVID demonstrate a similar economic effect to tariffs?",
            "What is the historical comparison made about Trump's tariffs?"
        ]
        # --- End of Predefined Questions ---

        # Run the tests
        run_predefined_tests(chatbot, engine, predefined_questions)

    elif args.mode == 'interactive':
         print("\n--- Starting Interactive Mode (Ctrl+C to exit) ---")
         conversation_history = [] # Keep track of history in interactive mode
         while True:
            try:
                question = input("\nAsk the chatbot a question: ")
                if question.lower() in ['exit', 'quit']:
                     break
                # Pass current history and get response
                response_data = chatbot.generate_response(question, engine, conversation_history=conversation_history)
                chatbot_response = response_data.get("chatbot_response", "Sorry, I encountered an issue.")

                print("\nRESPONSE:")
                print(chatbot_response)

                # Add interaction to history for the next turn
                conversation_history.append({
                     "user": question,
                     "chatbot_response": chatbot_response
                })
                # Optional: Prune history if it gets too long over many turns
                # MAX_HISTORY = 10 # Keep last 10 interactions
                # if len(conversation_history) > MAX_HISTORY:
                #      conversation_history = conversation_history[-MAX_HISTORY:]

            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                 print(f"\nAn error occurred: {e}")
                 # Optionally reset history or handle error more gracefully

# python chatbot.py --db_uri postgresql://admin:secret@localhost:5432/testdb --mode test --llm "deepseek-r1:14b"

# python chatbot.py --db_uri postgresql://admin:secret@localhost:5432/testdb --mode interactive --llm "deepseek-r1:7b"