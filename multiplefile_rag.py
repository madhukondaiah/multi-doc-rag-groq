import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# load env
load_dotenv(r"C:\Users\kandu\OneDrive\Desktop\LangChain\.env")

working_dir = os.path.dirname(os.path.abspath(__file__))

# embeddings
embedding = HuggingFaceEmbeddings()

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0
)

# ✅ PROCESS DOCUMENTS
def process_document_to_chroma_db(file_name):
    loader = PyMuPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    # ✅ Add correct metadata (IMPORTANT)
    for doc in documents:
        doc.metadata["source"] = file_name

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts_chunks = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=texts_chunks,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    vectordb.persist()

    return 0


# ✅ ANSWER QUESTION
def answer_question(user_question):
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    # ✅ MMR for diversity (FIXES SAME SOURCE ISSUE)
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain.invoke({
        "query": user_question + " Answer clearly and mention sources."
    })

    answer = response["result"]
    sources_docs = response["source_documents"]

    sources = []
    seen = set()

    for doc in sources_docs:
        file_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")

        entry = f"{file_name} (Page {page})"

        if entry not in seen:
            seen.add(entry)
            sources.append(entry)

    return answer, sources