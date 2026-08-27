# 🩺 MEDEASY

### AI-Powered Multi-Agent Medical Assistant

MEDEASY is an AI-powered healthcare assistance platform that combines multiple specialized AI agents to provide intelligent, context-aware responses to medical queries.

The system integrates Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), medical image analysis, scientific literature search, web search, and conversational AI into a unified multi-agent architecture.

> ⚠️ **Disclaimer:** MEDEASY is an educational and research project. It is not a medical diagnostic system and should not be used as a replacement for qualified healthcare professionals.

---

## ✨ Features

### 🤖 Multi-Agent AI Architecture

MEDEASY uses specialized AI agents rather than relying on a single model.

The system dynamically determines which agent is best suited to handle a user's request.

Supported agents include:

- 💬 Conversation Agent
- 📚 Medical RAG Agent
- 🌐 Web Search Agent
- 🫁 Chest X-Ray Analysis Agent
- 🧠 Brain Tumor Analysis Agent
- 🩹 Skin Lesion Segmentation Agent
- 🛡️ Human Validation / Safety Layer

---

### 💬 Conversational AI

MEDEASY can handle general medical conversations and follow-up questions using a conversational AI agent.

It can:

- Answer general medical questions
- Explain medical terminology
- Provide educational explanations
- Maintain conversational context
- Generate structured responses

---

### 📚 Retrieval-Augmented Generation

MEDEASY includes a medical knowledge retrieval system that allows the application to retrieve relevant information from its medical document collection before generating an answer.

This helps the system provide responses grounded in retrieved medical information instead of relying entirely on the language model's internal knowledge.

The RAG pipeline includes:

```text
Medical Documents
       ↓
Document Processing
       ↓
Text Chunking
       ↓
Embeddings
       ↓
Vector Database
       ↓
Relevant Document Retrieval
       ↓
LLM
       ↓
Final Response

System Architecture
                         ┌─────────────────────┐
                         │       User          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Backend   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Agent Decision    │
                         │      Router         │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │ Conversation│       │ Medical RAG │       │ Web Search  │
       │    Agent    │       │    Agent    │       │    Agent    │
       └─────────────┘       └──────┬──────┘       └─────────────┘
                                    │
                                    ▼
                             ┌─────────────┐
                             │   Qdrant    │
                             │Vector Store │
                             └─────────────┘

              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │ Chest X-Ray │       │ Brain Tumor  │       │ Skin Lesion │
       │    Agent    │       │    Agent     │       │    Agent    │
       └─────────────┘       └─────────────┘       └─────────────┘

                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Validation & Safety │
                         │      Layer          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   MEDEASY Response  │
                         └─────────────────────┘
