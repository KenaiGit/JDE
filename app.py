from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from helper import process_query
import os

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_product(request: Request):
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

# Entrypoint for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT",10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
