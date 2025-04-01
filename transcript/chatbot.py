from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
import psycopg2
import re

class Chatbot:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", llm_model="deepseek-r1:7b", llm_base_url="http://localhost:11434"):
        """
        Initialize the Chatbot with an embedding model and an LLM.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm = ChatOllama(model=self.llm_model, temperature=0.5, base_url=self.llm_base_url)
        
    def get_fragments_from_question(self, question, engine):
        """
        Extract fragments from the PostgreSQL table 'Clips' based on the embedding similarity.
        """
        emb = self.embedding_model.encode(
            question, 
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        # Convert the NumPy array to a string acceptable by pgvector
        embedding_str = "[" + ",".join(map(str, emb)) + "]"
        conn_str = engine.url.render_as_string(hide_password=False)
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        query = """
        SELECT clip_id, transcript, 
               1 - (embedding <-> %s) AS similarity
        FROM Clips
        ORDER BY embedding <-> %s
        LIMIT 2
        """
        cur.execute(query, (embedding_str, embedding_str))
        result = cur.fetchall()
        conn.close()
        return result
    
    def generate_response(self, user_question, engine, conversation_history=[]):
        """
        Generate a response by retrieving context from clips and feeding it into an LLM.
        
        This version adds conversation history into the prompt.
        It accumulates the most recent messages (starting from the newest) 
        until a 2,000-character context window is reached, then displays them in chronological order.
        """
        # Retrieve context fragments
        fragments = self.get_fragments_from_question(user_question, engine)
        context = [fragment[1] for fragment in fragments]
        documents = "\n\n".join(context)
        
        # Build conversation history string up to 2000 characters.
        # Iterate over conversation_history in reverse (most recent first)
        history_entries = []
        total_length = 0
        for entry in reversed(conversation_history):
            msg = f"User: {entry['user']}\nAssistant: {entry['chatbot_response']}\n{'-'*35}\n"
            if total_length + len(msg) > 20000:
                break
            history_entries.append(msg)
            total_length += len(msg)
        # Reverse back so that messages appear in chronological order
        conversation_history_str = "".join(reversed(history_entries)) if history_entries else "None"
        
        # Define the prompt including conversation history.
        prompt_template = PromptTemplate(
            template="""You are an AI assistant specialized in answering questions about context given, you have a conversation
            history so have that in mind.
Conversation History: {conversation_history}
Context: {documents}
Question: {user_question}
Answer:""",
            input_variables=["user_question", "documents", "conversation_history"]
        )
        rag_chain = prompt_template | self.llm | StrOutputParser()
        answer = rag_chain.invoke({
            "user_question": user_question,
            "documents": documents,
            "conversation_history": conversation_history_str,
        })
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        return {"chatbot_response": answer}

if __name__ == "__main__":
    # For standalone testing, you can run:
    # python3 chatbot.py --db_uri postgresql://admin:secret@localhost:5434/testdb
    import argparse
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description="Chatbot Standalone Mode")
    parser.add_argument("--db_uri", required=True, help="PostgreSQL connection URI (e.g., postgresql://user:pass@host/dbname)")
    args = parser.parse_args()
    engine = create_engine(args.db_uri)
    question = input("Ask the chatbot a question: ")
    response = Chatbot().generate_response(question, engine)
    print(response)
