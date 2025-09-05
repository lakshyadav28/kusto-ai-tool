from app.services.openai_service import AzureOpenAIService

class NL2KQL:
    def __init__(self, openai_service: AzureOpenAIService):
        self.openai_service = openai_service

    def nl_to_kql(self, user_query: str, context: str) -> str:
        print("Converting natural language to KQL:", user_query)

        # Construct prompt for OpenAI
        prompt = f"""
        You are a Kusto expert. Generate a KQL query for the following user request.

        Use the table named 'ServiceLogsTest' with columns:
        Time, EventId, Correlation, SID, Category, Message

        The context below lists relevant events (EventId and Message) for the user's query. The 'Tag' field in context is for your understanding only; it is not a table column. Use Category and EventId to filter the results as appropriate. For reliability or error-related queries, use EventId values from context and filter Category by the feature name.

        Example output format:
        ServiceLogsTest
        | where Category == "<feature_name>"
        | summarize errorCount = countif(EventId in (<event_ids>)) by bin(Time, 1d)

        Context:
        {context}

        Respond only with the KQL query. Do not include explanations or formatting.

        User query: "{user_query}"

        KQL:
        """
        kql_query = self.openai_service.execute(prompt)
        return kql_query