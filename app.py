from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from helper import process_query  # Importing your existing query processing function

# Initialize FastAPI app
app = FastAPI()

# Request body structure
class QueryRequest(BaseModel):
    query: str

# POST endpoint to handle user queries
from fastapi import FastAPI, HTTPException, Request
import uvicorn
from helper import process_query  # Your chatbot logic

app = FastAPI()

@app.post("/query/")
async def query_product(request: Request):
    """
    Accepts raw JSON input like: {"query": "your question here"}
    """
    try:
        body = await request.json()
        query = body.get("query")
        
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in request body.")
        
        result = process_query(query)
        return {"status": "success", "result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}


