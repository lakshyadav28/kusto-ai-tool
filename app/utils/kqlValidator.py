
from typing import List
from app.services.kusto_service import KustoService
from azure.kusto.data.exceptions import KustoServiceError
from app.services.openai_service import AzureOpenAIService

class KQLValidator:
    def __init__(self, known_columns: List[str], openai_service: AzureOpenAIService, kusto_service: KustoService):
        self.known_columns = known_columns

        self.openai_service = openai_service
        self.kusto_service = kusto_service

    def validate_and_run_kql(self, kql_query: str, context: str = "") -> dict:
        print("Validating KQL query...")
        result = self.kusto_service.run_query(kql_query)
        result_dict = result.to_dict(orient="records")[0] if not result.empty else {}
        success = result_dict.get("success", True)
        error = result_dict.get("error", None)
        if not success:
            if "semantic error" in error or error.lower().startswith("syntax error"):
                print("KQL query validation failed. Attempting to fix with LLM...")
                kql_query = self.fix_with_llm(kql_query, error)
                print("Fixed KQL query:", kql_query)
                return self.validate_and_run_kql(kql_query)
        return {"query": kql_query, "result": result}

    def fix_with_llm(self, original_query: str, error_message: str, context: str = "") -> str:
        print("Attempting to fix KQL query using LLM...")
        prompt = f"""
            You are a KQL expert. The following query has a semantic error:

            Original Query:
            {original_query}

            Error Message:
            {error_message}

            Context:
            {context}

            Please fix the query.
            Respond only with the fixed KQL query. Do not include explanations or formatting.

            KQL:
            """
        fixed_query = self.openai_service.execute(prompt)
        return fixed_query
