from langsmith import traceable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import SeleniumURLLoader

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

def rag_system():

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

    return vector_store

