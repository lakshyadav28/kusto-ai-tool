
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from app.utils.semantic_search import SemanticSearch

class AzureOpenAIService:
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str = "2023-07-01-preview"):
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2025-01-01-preview",
        )
        self.client = client
        self.deployment_name = deployment_name

    def execute(self, prompt: str) -> str:
        print("Executing prompt to generate chat response...")

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a KQL expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

        response = response.choices[0].message.content.strip().strip("`")
        return response
