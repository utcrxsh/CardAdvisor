# CardAdvisor

**Your Intelligent Credit Card Assistant - FastAPI + RAG + LangChain + Groq**

---

## What is CardAdvisor?

CardAdvisor is a blazing-fast, AI-powered credit card recommendation system for the Indian market. It leverages:

- **FastAPI** — Modern, high-performance Python web backend
- **RAG (Retrieval-Augmented Generation)** — Combines search and generation for accurate, context-aware answers
- **LangChain** — Orchestrates LLMs, retrieval, and tool use for advanced reasoning
- **Groq LLM** — Ultra-fast, cost-effective large language model API
- **FAISS** — Efficient vector search for instant retrieval from your credit card knowledge base

---

## Key Features

- **Personalized Recommendations** — Get the best credit card for your needs, instantly
- **Natural Language Chat** — Ask questions in plain English, get smart, context-rich answers
- **Card Comparison** — Side-by-side analysis of any two cards
- **Category Search** — Find top cards for travel, cashback, rewards, and more
- **Modern Web UI** — Clean, responsive chat interface (HTML + Bootstrap)
- **Debug Mode** — See behind-the-scenes agent reasoning (optional)

---

##  Tech Stack

- **Backend:** FastAPI (Python)
- **AI Engine:** LangChain (RAG, agent, tools)
- **LLM:** Groq API (Llama-3.3-70B)
- **Vector DB:** FAISS
- **Frontend:** HTML + Bootstrap + Vanilla JS

---

##  How It Works

1. **User asks a question** (e.g., “Best travel cards under ₹1000 annual fee?”)
2. **RAG pipeline** retrieves relevant card data using FAISS
3. **LangChain agent** decides which tools to use (lookup, compare, summarize, etc.)
4. **Groq LLM** generates a detailed, context-aware answer
5. **FastAPI** serves the response instantly to the web UI

---

##  Why CardAdvisor?

- **Lightning Fast** — Groq + FastAPI = instant answers
- **Smart** — Combines retrieval and reasoning for accuracy
- **Extensible** — Add new cards, categories, or tools easily
- **Production Ready** — Scalable, maintainable, and easy to deploy

---

## 👀 Demo

![Demo](demo.gif)

---

**Built with ❤️ using FastAPI, LangChain, RAG, and Groq.**