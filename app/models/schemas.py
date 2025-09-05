from pydantic import BaseModel
from typing import List, Optional, Dict

# Request model for chat input
class ChatRequest(BaseModel):
    query: str  # Natural language query from user
    context: Optional[Dict[str, str]] = None  # Optional context (e.g., feature name, time range)

# Response model for chat output
class ChatResponse(BaseModel):
    success: bool
    message: Optional[str] = None  # Summary or explanation
    data: Optional[List[Dict[str, str]]] = None  # Raw data from Kusto
    error: Optional[str] = None  # Error message if any
