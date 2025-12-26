# LexSum – AI-Powered Legal Document Summarization

LexSum is an **AI-driven legal document summarization system** designed to generate **accurate, concise, and context-aware summaries** of long and complex legal documents such as court judgments, petitions, and legal opinions.

The system leverages **LegalBERT embeddings combined with a Retrieval-Augmented Generation (RAG) pipeline** to ensure **high factual consistency, zero hallucination, and strong legal relevance**.

---

## Problem Statement

Legal documents are:

* Extremely long and complex
* Written in dense legal language
* Time-consuming to read and analyze manually

Generic summarization models fail to capture legal nuances and may generate **hallucinated or legally incorrect content**, which is unacceptable in the legal domain.

---

## 💡 Solution Overview

LexSum addresses this problem by using:

* **LegalBERT** for domain-specific semantic understanding
* **RAG (Retrieval-Augmented Generation)** for extracting only the most legally relevant sentences
* **Extractive summarization** to preserve original legal wording and accuracy

---

## 🧠 Key Features

* Upload any legal PDF (judgment, petition, order, etc.)
* Domain-specific semantic understanding using LegalBERT
* RAG-based sentence retrieval (no hallucinations)
* Extractive summaries with preserved legal meaning
* Supports long legal documents
* High factual consistency and legal keyword coverage

---

## 🏗️ System Architecture (High Level)

```
Legal PDF Upload
      ↓
Text Extraction & Preprocessing
      ↓
Sentence Embedding (LegalBERT)
      ↓
Vector Indexing (FAISS)
      ↓
RAG-Based Sentence Retrieval
      ↓
Sentence Ranking
      ↓
Final Extractive Legal Summary
```

---

## ⚙️ How It Works (Step-by-Step)

1. User uploads a legal PDF document
2. Text is extracted, cleaned, and split into sentences
3. Each sentence is converted into embeddings using **LegalBERT**
4. Embeddings are stored in a **FAISS vector index**
5. Relevant sentences are retrieved using **cosine similarity**
6. Sentences are ranked using semantic relevance, keywords, and position
7. Top-ranked sentences are assembled into a concise extractive summary

---

## 🚀 Why RAG?

* Prevents hallucination (critical for legal documents)
* Ensures all summary content comes from the source document
* Handles very long documents efficiently
* Preserves factual and legal correctness
---

## 🧩 Why LegalBERT?

* Trained specifically on legal corpora (judgments, statutes, case laws)
* Understands legal terminology and judicial reasoning
* Outperforms general-purpose language models in legal NLP tasks
* Produces embeddings aligned with legal semantics
---

## 🛠️ Tech Stack

* **Language:** Python
* **Models:** LegalBERT
* **NLP:** HuggingFace Transformers
* **Vector Search:** FAISS
* **PDF Processing:** pdfplumber / PyMuPDF
* **Similarity Metric:** Cosine Similarity



Just tell me 👍
