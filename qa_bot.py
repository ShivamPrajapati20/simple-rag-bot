"""
    qa_bot.py
    A simple Q&A bot that reads a text file and answers questions about it.
    Usage: python qa_bot.py my_document.txt
"""
import sys, anthropic, numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = SentenceTransformer("all-MiniLM-L6-v2")

#1: read and chunk the text file
def read_and_chunk(file_path: str, chunk_size: int = 200) -> list:
    """
        Read a text file and split it into chunks.
        chunk_size = number of words per chunk.
    """
    with open(file_path, "r") as f:
        text = f.read()

    # split into words, then group into chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    print(f"Split '{file_path}' into {len(chunks)} chunks.")
    return chunks


#2: embed all chunks
def get_embedding(text: str) -> list:
    return model.encode(text).tolist()

def embed_chunks(chunks: list) -> list:
    """Embed every chunk. Returns a list of (chunk_text, embedding) pairs."""
    print("Embedding chunks... (this takes a few seconds)")
    embedded = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        embedded.append((chunk, emb))
        print(f"  {i+1}/{len(chunks)} done", end="\r")
    print(f"\nAll {len(embedded)} chunks embedded.")
    return embedded


#3: search
def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search(question: str, embedded_chunks: list, top_k: int = 3) -> list:
    """Find the top_k most relevant chunks for the question."""
    q_emb = get_embedding(question)
    scores = []
    for chunk_text, chunk_emb in embedded_chunks:
        score = cosine_similarity(q_emb, chunk_emb)
        scores.append((score, chunk_text))
    scores.sort(reverse=True)
    return scores[:top_k]


#4: ask Claude
def ask(question: str, embedded_chunks: list) -> str:
    """Search for relevant context, then ask Claude to answer."""
    results = search(question, embedded_chunks, top_k=3)
    context = "\n\n".join([chunk for score, chunk in results])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system="""You are a helpful assistant. Answer the question using ONLY
                the context provided below. If the answer is not in the context,
                say "I could not find that in the document."
                Do not use outside knowledge.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qa_bot.py my_document.txt")
        sys.exit(1)

    file_path = sys.argv[1]

    # Load and embed the document
    chunks = read_and_chunk(file_path)
    embedded_chunks = embed_chunks(chunks)

    print(f"\nReady! Ask questions about '{file_path}'.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        answer = ask(question, embedded_chunks)
        print(f"\nAnswer: {answer}\n")