import os
from fastapi import FastAPI, HTTPException, Request
from helper import process_query
import uvicorn
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

        # Lazy import for memory optimization
         
        result = process_query(query)

        return {"status": "success", "result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
@app.get("/")
async def root():
    return {"message": "Welcome to the Query API! Try /query/ or /health/"}



if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

