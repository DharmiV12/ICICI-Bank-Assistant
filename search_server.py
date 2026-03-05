from fastmcp import FastMCP
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma

mcp = FastMCP(name="searching_tools")


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(embedding_function=embedding,persist_directory="./chroma")


@mcp.tool(name="rag_tool", description="Retrive the relevent documents from content related to query")
async def rag_tool(query: str):
    """
    Retrieve the relevent information from the PDF document
    Use this tool when user ask ICICI Bank realed question
    """

    print("\n\nRAG Tool is called\n\n")

    retriver = vector_store.similarity_search_with_score(query=query, k=3)

    if not retriver:
        return "No Matched Documnet Found"

    print("retriver: ", retriver, "\n\n")     

    content = "\n\n".join(
        (f"\nType of doc: {type(doc)}\nContent: {doc[0].page_content}")
        for doc in retriver
    )

    print("source and content printed.\n")
    print("Content: ", content, "\n\n")
    print("Tool gave answer correctly\n")

    return content


@mcp.tool(name="duckduckgo_search", description="it finds result from web and give answer of query")
async def duckduckgo_search(query: str):
    """
    Use when rag_tool fails and it gives "No Matched Documnet Found".
    """

    print("\nWeb Search tool is called\n\n")

    search_tool = DuckDuckGoSearchRun()

    return search_tool.run(query)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8005)