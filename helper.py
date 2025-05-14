import os
from dotenv import load_dotenv
from few_shots import few_shots  # your examples

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_together import Together
from langchain.prompts import FewShotPromptTemplate, SemanticSimilarityExampleSelector, PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def reset_session_state():
    print("🔄 Resetting session state...")

def process_query(query):
    print(f"🔍 Processing query: {query}")
    result = "Unable to process the query."

    try:
        # Load environment variables
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")
        DB_PORT = os.getenv("DB_PORT")
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

        # Step 1: Connect to PostgreSQL
        db = SQLDatabase.from_uri(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
            sample_rows_in_table_info=3
        )

        # Step 2: Initialize LLM
        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            temperature=0.2,
            max_tokens=500,
            together_api_key=TOGETHER_API_KEY,
        )

        # Step 3: Create embeddings & vectorstore (in memory only)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_texts = [" ".join(str(value) for value in example.values()) for example in few_shots]
        vectorstore = Chroma.from_texts(
            texts=vector_texts,
            embedding=embeddings,
            metadatas=few_shots
        )

        # Step 4: Create semantic selector
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=2
        )

        # Step 5: Build the prompt
        postgres_prompt = """You are a PostgreSQL expert. Given an input question, write a syntactically correct SQL query to run, then return the result.

Instructions:
- Never use SELECT *.
- Only include the columns necessary to answer the question.
- Use LIMIT {top_k} where it makes sense.
- Avoid filtering by date unless the question explicitly mentions a time frame (e.g., "today", "last week", "on April 1st").
- Focus on clarity and correctness.
- Do not use aliases unless they improve readability or are required.

Table information:
- All data resides in a single table. Use column names directly.
- Infer filter conditions and sort order based on the user's natural language question.
"""

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

        # Step 6: Create the SQL chain
        chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            verbose=False,
            prompt=few_shot_prompt,
            use_query_checker=True,
            return_intermediate_steps=False
        )

        # Step 7: Run the chain
        response = chain.invoke({"query": query})
        result = response.get("result", "No result returned.")
        print(f"✅ Answer: {result}")

    except Exception as e:
        result = f"❌ Error: {e}"
        print(result)

    reset_session_state()
    return result
