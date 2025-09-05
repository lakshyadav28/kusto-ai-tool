
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.kusto_service import KustoService
from app.services.openai_service import AzureOpenAIService

router = APIRouter()

# Initialize services
kusto_service = KustoService(
    cluster_url="<cluster_url>",
    database="TestDB"
)

openai_service = AzureOpenAIService(
    endpoint="<endPoint>",
    deployment_name="<name>",  # e.g., "gpt-35-turbo"
    api_version="2025-01-01-preview"
)

@router.post("/chat", response_model=ChatResponse)
async def chat_with_kusto(request: ChatRequest):
    try:
        # Step 1: Convert NL to KQL
        print("Converting natural language to KQL:", request.query)
        kql_query = openai_service.nl_to_kql(request.query)

        # Step 2: Run KQL on ADX
        print("Running KQL query on Azure Data Explorer...")
        result = kusto_service.run_query(kql_query)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])

        # Step 3: Return structured response
        return ChatResponse(
            success=True,
            message=f"KQL executed: {kql_query}",
            data=result["data"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
