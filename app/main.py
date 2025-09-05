
from fastapi import FastAPI
from app.routes import chat

app = FastAPI(
    title="Kusto AI Log Analyzer",
    description="Chat-based interface to analyze Kusto logs using natural language",
    version="1.0.0"
)

# Include chat route
app.include_router(chat.router, prefix="/api", tags=["Chat"])

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Kusto AI Log Analyzer is running"}
