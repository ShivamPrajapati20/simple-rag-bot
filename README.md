# Simple RAG Bot — Week 9

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-orange)
![Sentence Transformers](https://img.shields.io/badge/Embeddings-sentence--transformers-green)
![Anthropic](https://img.shields.io/badge/Powered%20by-Claude%20Haiku-purple)
![License](https://img.shields.io/badge/license-MIT-orange)

A RAG (Retrieval-Augmented Generation) Q&A bot built **from scratch** — no vector database libraries, no LangChain. Just Python, NumPy, and the Anthropic API.

Ask questions about any `.txt` file and get answers grounded in that document.

---

## What it does

1. Reads any `.txt` file and splits it into chunks
2. Converts each chunk into an embedding (a list of numbers representing meaning)
3. When you ask a question, finds the most relevant chunks using cosine similarity
4. Passes those chunks to Claude as context
5. Claude answers using only what was found in the document

---

## Why build it from scratch?

Most tutorials jump straight to ChromaDB or LangChain. Building it manually first means you understand exactly what every vector database does under the hood — embeddings, cosine similarity, retrieval, grounding. Week 10 uses ChromaDB and the concepts are immediately obvious because of this week.

---

## What I learned building this

- What an **embedding** is — a list of numbers that represents the meaning of text
- How **cosine similarity** works — measuring how close two embeddings are (1.0 = same meaning, 0.0 = unrelated)
- Why **RAG** is better than stuffing a whole document into the prompt — search first, then answer
- How to **ground** Claude's answers to a specific document using the system prompt
- The limitation of in-memory storage — and why vector databases like ChromaDB exist (Week 10)

---

## Demo

```
$ python qa_bot.py python_notes.txt

Split 'python_notes.txt' into 4 chunks.
Embedding chunks... (this takes a few seconds)
All 4 chunks embedded.

Ready! Ask questions about 'python_notes.txt'.
Type 'quit' to exit.

Your question: How do I install a Python package?

Answer: To install a Python package, use the command:
        pip install package-name

Your question: What is the capital of Japan?

Answer: I could not find that in the document.
```

---

## Setup

### Prerequisites
- Python 3.11+
- Anthropic API key — free at [console.anthropic.com](https://console.anthropic.com)

### Installation

```bash
git clone https://github.com/ShivamPrajapati20/simple-rag-bot.git
cd simple-rag-bot

pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

### Run it

```bash
python qa_bot.py your_document.txt
```

### Try it with the sample file

```bash
python qa_bot.py sample.txt
```

---

## How it works — the 4 steps

```
your_document.txt
      ↓
Step 1: Split into chunks (200 words each)
      ↓
Step 2: Embed each chunk using sentence-transformers
      ↓
Step 3: User asks a question → embed question → cosine similarity → top 3 chunks
      ↓
Step 4: Give chunks + question to Claude → Claude answers from context only
```

---

## Project structure

```
simple-rag-bot/
├── qa_bot.py          ← the complete bot (all logic in one file)
├── sample.txt         ← a sample document to test with
├── requirements.txt
├── .env.example       ← ANTHROPIC_API_KEY=your-key-here
├── .gitignore
└── README.md
```

---

## Known limitations

- **No persistence** — embeddings are stored in memory only. Close the script and they are gone. Re-embedding happens every time you start.
- **Simple chunking** — splits by word count, not by sentence or paragraph. A chunk may cut mid-sentence.
- **Slow at scale** — linear search across all chunks. Fine for small documents, slow for 10,000+ chunks.

These limitations are exactly what [Week 10 (ChromaDB)](https://github.com/ShivamPrajapati20/pdf-rag-bot) fixes.

---

## Tech stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| sentence-transformers | Local embeddings — free, no API key |
| NumPy | Cosine similarity calculation |
| Anthropic SDK | Claude API for answering questions |
| python-dotenv | Secure API key loading |

---

## Part of my AI Engineering roadmap

This is Project 7 of my AI/LLM Engineering learning journey — Phase 3, Week 9.

**Previous projects:**
- [AI CLI Chatbot](https://github.com/ShivamPrajapati20/ai-cli-chatbot)
- [AI Data Analyser](https://github.com/ShivamPrajapati20/ai-data-analyser)
- [Prompt Library](https://github.com/ShivamPrajapati20/prompt-library)
- [AI JD Pipeline](https://github.com/ShivamPrajapati20/ai-jd-pipeline)
- [LLM Benchmark](https://github.com/ShivamPrajapati20/llm-benchmark)
- [Resume Analyser](https://github.com/ShivamPrajapati20/resume-analyser)

**Next:** [Week 10 — ChromaDB persistent vector storage](https://github.com/ShivamPrajapati20/pdf-rag-bot)

---

## License

MIT
