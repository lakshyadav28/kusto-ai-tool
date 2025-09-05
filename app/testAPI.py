from fastapi import HTTPException

from app.models.schemas import ChatResponse
from app.services.openai_service import AzureOpenAIService
from app.services.kusto_service import KustoService
from app.utils.embedNew import EmbeddingGenerator
from app.utils.embeddingSearcher import EmbeddingSearcher
from app.utils.kqlValidator import KQLValidator
from app.utils.nl2kql import NL2KQL
from app.utils.semantic_search import SemanticSearch

if __name__ == "__main__":
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

    #semantic_search = SemanticSearch("kusto_knowledge.index", "kusto_knowledge.json")
    embeddingSystem = EmbeddingGenerator()
    
    nl2kql = NL2KQL(openai_service)

    validator = KQLValidator(known_columns=["Time", "EventId", "Correlation", "SID", "Category", "Message"], openai_service=openai_service, kusto_service=kusto_service)

    # Step 1: Convert NL to KQL
    print("Enter your query (e.g., 'Show average TimeTaken for failures'):")
    user_input = input(">> ")

    #step 2: Generate embedding
    embeddingSystem.generate_and_save_embeddings("metadata.json")

    # Step 2: Extract context
    embeddingSearcher = EmbeddingSearcher()
    # Get LLM context
    context = embeddingSearcher.prepare_llm_context(user_input)
    if not context:
        print("No relevant items found for context.")
    else:
        print(context)

        # Step 3: Generate KQL query
        kql_query = nl2kql.nl_to_kql(user_input, context)

        # Step 3.1: Validate and fix KQL query
        validation_result = validator.validate_and_run_kql(kql_query, context)
        kql_query = validation_result.get("query", kql_query)
        print("Validated KQL query:", kql_query)
        result = validation_result.get("result", {})

        # Step 5: Check for errors in the result
        # Extract success and error from result DataFrame or dict
        if hasattr(result, "to_dict"):  # Check if result is a DataFrame
            result_dict = result.to_dict(orient="records")[0] if not result.empty else {}
            success = result_dict.get("success", True)
            error = result_dict.get("error", None)
        else:
            success = False
            error = None
        if not success:
            error_detail = error or "Unknown error occurred while running KQL query."
            print(f"Error: {error_detail}")
        else:
            # Return structured response
            print("Response received from Kusto:", result.get("data", result) if isinstance(result, dict) else result)
            