"""
    pdf_qa_langchain.py
    PDF Q&A bot using LangChain + ChromaDB + Claude.
    Usage: python pdf_qa_langchain.py your_document.pdf
"""
import sys, os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def build_vectorstore(pdf_path: str):
    """Load PDF into ChromaDB. Skips if data already exists."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # If ChromaDB already has data, just load it from disk
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing ChromaDB from disk...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Otherwise load the PDF and create ChromaDB from scratch
    print(f"Loading '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print("Done.")
    return vectorstore


def build_qa_chain(vectorstore):
    """Build the RetrievalQA chain."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=400)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )


def ask(qa_chain, question: str):
    """Ask a question and print the answer with page citations."""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    pages = sorted(set(
        str(d.metadata.get("page", 0) + 1)
        for d in result["source_documents"]
    ))
    print(f"\nAnswer: {answer}")
    print(f"Source: page(s) {', '.join(pages)}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_qa_langchain.py your_document.pdf")
        sys.exit(1)

    vectorstore = build_vectorstore(sys.argv[1])
    qa_chain = build_qa_chain(vectorstore)

    print(f"\nReady! Ask questions about '{sys.argv[1]}'.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit": break
        if question: ask(qa_chain, question)