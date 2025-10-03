from typing import Dict, Any, Optional, Union, List, Tuple
from config.setting import env
import clickhouse_connect

SYNC_DB_URL = (
    f"clickhouse://{env.clickhouse_user}:{env.clickhouse_password}@"
    f"{env.clickhouse_host}:{env.clickhouse_http_port}/{env.clickhouse_database}"
)

class ClickhouseDb:
    def __init__(self):
        self.client = None
        
    async def _ensure_client_initialized(self):
        """
        An internal method to create the async client if it doesn't exist.
        This is called before every query.
        """
        if self.client is None:
            try:
                self.client = await clickhouse_connect.create_async_client(
                    host=env.clickhouse_host,
                    port=env.clickhouse_http_port,
                    username=env.clickhouse_user,
                    password=env.clickhouse_password,
                    database=env.clickhouse_database
                )
            except Exception as e:
                raise e
        
    async def aexecute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        retrieve_header: bool = False
    ) -> Optional[Union[List[tuple], Tuple[List[str], List[tuple]]]]:
        """
        Asynchronously executes a SQL query with parameters and optional headers.

        Args:
            query: The SQL query string with named parameters (e.g., %(my_param)s).
            params: A dictionary of parameters to bind to the query.
            retrieve_header: If True, returns a tuple of (headers, rows).
                               Otherwise, returns only the rows.

        Returns:
            The query results, or None if an error occurs.
        """
        try:
            await self._ensure_client_initialized()

            result = await self.client.query(query, parameters=params)
            
            if retrieve_header:
                headers = result.column_names
                rows = result.result_rows
                return headers, rows
            
            return result.result_rows
            
        except Exception as e:
            return e

db = ClickhouseDb()

if __name__ == "__main__":
    import asyncio
    async def main():
        """Main function to test the ClickhouseDb class."""
        await db.execute_query(
            """
            CREATE TABLE IF NOT EXISTS HopeReportCancelledBill
            (
                CancelDate DateTime,
                Amount Float64
            )
            ENGINE = MergeTree()
            ORDER BY CancelDate
            """
        )

        test_query = """
        SELECT
            count(*)
        FROM
            HopeReportCancelledBill 
        WHERE
            toDateTime(CancelDate) >= %(start_date)s
            AND toDateTime(CancelDate) <= %(end_date)s
        """
        
        params = {
            "start_date": "2023-02-01 00:00:00",
            "end_date": "2024-02-28 23:59:59"
        }
        
        # --- Test Case 1: Get results without headers ---
        print("\n--- Testing query without headers ---")
        result_rows = await db.aexecute_query(test_query, params)
        if result_rows is not None:
            print("Async query result (rows only):")
            print(result_rows)

        # --- Test Case 2: Get results with headers ---
        print("\n--- Testing query with headers ---")
        result_with_headers = await db.aexecute_query(test_query, params, retrieve_header=True)
        if result_with_headers is not None:
            headers, rows = result_with_headers
            print("Async query result (with headers):")
            print("Headers:", headers)
            print("Rows:", rows)
    asyncio.run(main())
