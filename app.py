from fastapi import FastAPI, HTTPException, Request
import uvicorn
import os
from helper import process_query  # Import your query handling logic

# Initialize FastAPI app
app = FastAPI()

# Health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "healthy"}

# POST endpoint to handle user queries
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

# Entry point for local testing (Render ignores this block)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render-assigned port or fallback to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
