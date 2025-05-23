import os
from fastapi import FastAPI, HTTPException, Request
from helper import process_query
 
app = FastAPI()
 
# Root endpoint supports GET and HEAD, hidden from docs
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    return {"message": "API is up and running"}
 
# Health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
 
# POST /query/ endpoint that expects JSON with a 'query' field
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
 
        # Lazy import for performance reasons
        
        result = process_query(query)
 
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)