from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from dotenv import load_dotenv
import os
import logging


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    filename="agents_log.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=API_KEY ,model="llama-3.3-70b-versatile", temperature=0)

app = FastAPI()

SERVERS = {
    "search_server": {
        "transport":"streamable-http",
        "url": "http://localhost:8005/mcp"
    }   
}

initial_prompt = """

You are an intation checker.

Check user question and detect intation from below labels:
1. GREETINGS / CASUAL TALKS
2. ICICI BANK RELATED QUERY
3. OUT-OF-DOMAIN QUERY

Check and return one label from this.

Label:

If Label is 1. GREETINGS / CASUAL TALKS:
- Give response according to you.

If Label is 2. ICICI BANK RELATED QUERY, 
- Give the answer using rag_tool, if you don't get answer from rag_tool then and then use search_tool
- Give deatiled answer of the question.
- Only answer the ICICI Bank related question. 
- With answer also return from which tool you get answer, like this tool_used = tool name
- If you don't get response from any of the tool, then don't create you own answer.
= Don't give extra answer.

If Label is 3. OUT-OF-DOMAIN QUERY:
-  Return "I can help with ICICI Bank services and banking queries. Please ask a related question."

Provide result accurate and in format and don't give "\n" and all things in answer.
"Don't give Labels in result."
If query is unclear then ask for clarification.
"""

@app.post('/question-answer/')
async def query_answer(query: str):

    try:

        client = MultiServerMCPClient(SERVERS)

        tools = await client.get_tools()

        print(f"Tools are {tools}\n")

        logging.info("Agent making proccess called")

        print("App called...\n\n")

        prompt = SystemMessage(
            content=[
                {
                    "type":"text",
                    "text":initial_prompt
                }
            ]
        )

        logging.info("Tools are binded")

        agent = create_agent(model=llm, tools=tools, system_prompt=prompt, debug=True)
        print("Agent created \n")

        logging.info("Agent created")

        response = await agent.ainvoke({"messages": query})

        logging.info("Successfully agent invoked and got result")

        print("Successfully get result from tools")

        print("Response: ", response, "\n")

        result = response["messages"][-1].content

        print("result: ", result)

        return result
    
    except Exception as e:

        logging.warning(f"During agent creation error occured, Error: {e}")

        print("Error: ", e)

        return {"Error": e}