import os
from dotenv import load_dotenv
from functools import lru_cache

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain.prompts import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain_together import Together
from few_shots import few_shots  # Ensure this exists

# Load environment variables
load_dotenv()

# Constants
DB_URI = "postgresql+psycopg2://postgres:root@localhost:5432/inventory"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

@lru_cache(maxsize=1)
def get_few_shot_db_chain():
    print("🔄 Initializing chain and dependencies...")
    
    # Connect to DB
    try:
        db = SQLDatabase.from_uri(DB_URI, sample_rows_in_table_info=3)
        print("✅ PostgreSQL connected!")
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return None

    # Load LLM
    llm = Together(
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=500,
        together_api_key=os.getenv("TOGETHER_API_KEY"),
    )

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Vector texts
    vector_texts = [" ".join(str(value) for value in ex.values()) for ex in few_shots]
    vectorstore = Chroma.from_texts(
        texts=vector_texts,
        embedding=embeddings,
        metadatas=few_shots
    )

    # Example selector
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2
    )

    # Prompt template
    postgres_prompt = """You are a PostgreSQL expert..."""  # Keep your original instructions here

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=postgres_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )

    # Final chain
    chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=False,
        prompt=few_shot_prompt,
        use_query_checker=True,
        return_intermediate_steps=False
    )
    print("✅ SQLDatabaseChain ready!")
    return chain

# Process query
def process_query(query):
    print(f"🔍 Processing: {query}")
    chain = get_few_shot_db_chain()
    if not chain:
        return "❌ Failed to initialize the chain."

    try:
        response = chain.invoke({"query": query})
        result = response.get("result", "No result returned.")
        return str(result) if isinstance(result, dict) else result
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return f"❌ Error processing query: {e}"
