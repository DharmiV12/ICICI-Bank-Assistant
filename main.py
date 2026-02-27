from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langsmith import traceable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import SeleniumURLLoader
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

API_KEY = os.getenv("GOQ_API_KEY")

llm = ChatGroq(api_key=API_KEY ,model="llama-3.3-70b-versatile", temperature=0)

app = FastAPI()

urls = [
    "https://www.icici.bank.in/personal-banking/deposits/fixed-deposit/fixed-deposit-faqs?ITM=nli_upi_payments_moneyTransfer_upi_cardjourney_1CTA_CMS_iciciBankNriWorldDebitCard_apply_NLI",
    "https://www.icici.bank.in/personal-banking/accounts/savings-account/savings-account-faqs?ITM=nli_help_help_entryPoints_1_CMS_savingsAccountFaqs_NLI",
    "https://www.icici.bank.in/personal-banking/deposits/fixed-deposit/tax-saver-fd/faqs",
    "https://www.icici.bank.in/personal-banking/deposits/recurring-deposits/recurring-deposit-faqs?ITM=nli_help_help_entryPoints_1_CMS_recurringDepositFaqs_NLI",
    "https://www.icici.bank.in/personal-banking/payments/bill-payments-and-recharges/aadhaar-pay/faqs",
    "https://www.icici.bank.in/personal-banking/payments/bill-payments-and-recharges/faqs?ITM=nli_help_help_entryPoints_1_CMS_billPaymentFaqs_NLI",
    "https://www.icici.bank.in/personal-banking/payments/bill-payments-and-recharges/online-recharge/faqs?ITM=nli_help_help_entryPoints_2_CMS_onlineRechargeFaqs_NLI",
    "https://www.icici.bank.in/personal-banking/payments/fastag/fastag-faqs",
    "https://www.icici.bank.in/personal-banking/cards/credit-card/faqs?ITM=nli_home_na_loanjourney_1CTA_CMS_homeLoan_apply_NLI",
    "https://www.icici.bank.in/personal-banking/cards/debit-card/debit-cards-faqs?ITM=nli_help_help_entryPoints_1_CMS_debitCardFaqs_NLI",
    "https://www.icici.bank.in/personal-banking/cards/travel-card/travel-cards-faqs?ITM=nli_home_na_loanjourney_2CTA_CMS_personalLoan_details_NLI",
    "https://www.icici.bank.in/personal-banking/cards/prepaid-card/prepaid-card-faqs?ITM=nli_help_help_entryPoints_1_CMS_prepaidCardFaqs_NLI",
    ]

@traceable(name="load_urls")
def load_urls(urls: list):
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    print("URL Data: ", data)
    return data

@traceable(name="split_documents")
def split_documents(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(data)
    print("length of chunks: ", len(chunks), "\n\n")
    return chunks

@traceable(name="build_vectorstore")
def build_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding: ", embedding, "\n\n")

    vector_store = Chroma(embedding_function=embedding, persist_directory="./chroma")
    add_docs = vector_store.add_documents(documents=chunks)

    print("vector store: ", vector_store, "\n\n")

    return vector_store

@traceable(name="setup_pipeline")
def setup_pipeline(urls: list):
    data = load_urls(urls)
    chunks = split_documents(data)
    vs = build_vectorstore(chunks)
    return vs

vector_store = setup_pipeline(urls)


search_tool = DuckDuckGoSearchRun()

@tool
def web_search_tool(query: str):
    """
    Use when rag_tool fails and it gives "No Matched Documnet Found".
    """

    logging.info("Web Search tool is used")

    print("\nWeb Search tool is called\n\n")

    return search_tool.run(query)


@tool
def rag_tool(query: str):
    """
    Retrieve the relevent information from the PDF document
    Use this tool when user ask ICICI Bank realed question
    """

    logging.info("Rag tool is used")

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

Provide result accurate and in format.
"Don't give Labels in result."
If query is unclear then ask for clarification.
"""

@app.post('/question-answer/')
def query_answer(query: str):

    try:

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
        
        tools = [web_search_tool, rag_tool]
        print("List of tools: ", tools, "\n")

        llm_with_tool = llm.bind_tools(tools)
        print(f"Binding tools: {llm_with_tool}", "\n")

        logging.info("Tools are binded")

        agent = create_agent(model=llm, tools=tools, system_prompt=prompt, debug=True)
        print("Agent created \n")

        logging.info("Agent created")

        response = agent.invoke({"messages": query})

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