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

        print("📥 Received request body:", body)  # DEBUG LOG
        if not query:
            print("⚠️ Missing 'query' in request body")
            raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

        print("🔍 Processing query:", query)
        result = process_query(query)
        print("✅ Query result:", result)

        return {"status": "success", "result": result}

    except Exception as e:
        print("❌ Error occurred:", str(e))  # DEBUG LOG
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
@app.get("/")
async def root():
    return {"message": "API is alive and kicking!"}


# Entrypoint for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT",10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
