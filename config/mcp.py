from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_search_suggestion = MultiServerMCPClient({
    "search-suggestion": {
        "url": "https://uat-genai-soap.siloamhospitals.com/suggestion/mcp",
        "transport": "streamable_http"
    }
})

