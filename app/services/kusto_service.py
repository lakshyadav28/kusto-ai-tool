
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
import traceback
import pandas as pd

class KustoService:
    def __init__(self, cluster_url: str, database: str):
        self.cluster_url = cluster_url
        self.database = database
        self.kcsb = KustoConnectionStringBuilder.with_interactive_login(cluster_url)
        self.client = KustoClient(self.kcsb)

    def run_query(self, kql: str) -> pd.DataFrame:
        try:
            print("Running KQL query on Azure Data Explorer...")
            response = self.client.execute(self.database, kql)
            if response.primary_results[0].rows_count == 0:
                return pd.DataFrame({"success": [False], "error": ["No rows returned. Check your query syntax and ensure it returns a tabular result."]})
            df = pd.DataFrame([row.to_dict() for row in response.primary_results[0]])
            return df
        except KustoServiceError as e:
            print("KustoServiceError occurred while running KQL query:", e)
            return pd.DataFrame({"success": [False], "error": [e]})
        except Exception as e:
            print("Error occurred while running KQL query:", e)
            error_message = f"{str(e)}\n{traceback.format_exc()}"
            return pd.DataFrame({"success": [False], "error": [error_message]})
