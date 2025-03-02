from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
import argparse
from sentence_transformers import SentenceTransformer
import requests
from sqlalchemy import create_engine, text
import psycopg2
import re

from collections import Counter

class Chatbot:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", llm_model="deepseek-r1:14b", llm_base_url="http://localhost:11434"):
        """
        Initialize the Chatbot class.

        :param course_code_to_name: Dictionary mapping course codes to their respective names.
        :param embedding_model: The model used for generating embeddings.
        :param llm_model: The LLM model used for generating responses.
        :param llm_base_url: The base URL for the LLM API.
        """

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension() # dimensions of embedding


        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm = ChatOllama(model=self.llm_model, temperature=0.5, base_url=self.llm_base_url)
        
       

        # self.api_base_url = "https://datastic4-4d0e75205d73.herokuapp.com/datastic_4/"
        # self.api_base_url = "http://127.0.0.1:5000/datastic_4/"


    def get_fragments_from_question(self, question, engine):
        """
        Extract fragments relevant to the question using the embedding model.

        :param question: The user's question.
        :param embedding: The embedding model instance.
        :return: Relevant fragments.
        """
        emb = self.embedding_model.encode(
            question, 
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        # 2) Convert that NumPy array to a pgvector-friendly string
        embedding_str = "[" + ",".join(map(str, emb)) + "]"

        # 3) Execute query with placeholders
        conn_str = engine.url.render_as_string(hide_password=False)
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        # print(f"\nThe embedding of the question has type of: {type(emb)} with length of: {len(emb)} and it is: {emb}\n")
        headers = {'Content-Type': 'application/json'}
        # data = {"emb" : str(emb)}


        query = "SELECT * from Clips"
        
        cur.execute("""SELECT clip_id, transcript, 
                1 - (embedding <-> %s) AS similarity
        FROM Clips
        ORDER BY embedding <-> %s
        LIMIT 2""", (embedding_str, embedding_str, ))
        result = cur.fetchall()
        conn.close()
        return result
    
    def generate_response(self, user_question, engine, conversation_history=[]):
        """
        Generates a response based on the user's question.

        :param request: A Flask request object containing JSON input.
        :return: A response dictionary with chatbot response, conversation history, and current course.
        """

        # embedding = OllamaEmbeddings(model=self.embedding_model)
    
        
        fragments = self.get_fragments_from_question(user_question, engine)

        context = [f[1] for f in fragments]
        print(f"The retrieved context is: {context}")
        documents = "\n\n".join(c for c in context)
        conversation_history_str = "\n".join(
            f"user_question: '{entry['user']}'\n\nchatbot_response: '{entry['chatbot_response']}'\n{'-'*35}"
            for entry in conversation_history
        ) if conversation_history else "None\n"

        prompt_template = PromptTemplate(
            template="""You are an AI assistant specialized in answering questions about college courses. 
            
            Context: {documents}
            Question: {user_question}
            Answer:""", #Conversation History: {conversation_history}
            input_variables=["user_question", "documents"] #, "conversation_history"]
        )

        rag_chain = prompt_template | self.llm | StrOutputParser()
        answer = rag_chain.invoke({
            "user_question": user_question,
            "documents": documents,
            # "conversation_history": conversation_history_str
        })
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        # conversation_history.append({'user': user_question, 'chatbot_response': answer})

        return {
            "chatbot_response": answer,
            # "conversation_history": conversation_history,
        }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data into PostgreSQL with pgvector")

    parser.add_argument("--db_uri", required=True,
                        help="PostgreSQL connection URI (postgresql://user:pass@host/dbname)")
    
    args = parser.parse_args()

    engine = create_engine(args.db_uri)
    question = "How much is the proposed tariff on China?" #input("Ask the chatbot a question: ")

    print(f"The question is: {question}")
 

    while 1:
        response = Chatbot().generate_response(question,engine)
        
        print(response)
        question = input("\nAsk the chatbot a question: ")

# usage: python3 chatbot.py --db_uri postgresql://admin:secret@localhost:5432/testdb



