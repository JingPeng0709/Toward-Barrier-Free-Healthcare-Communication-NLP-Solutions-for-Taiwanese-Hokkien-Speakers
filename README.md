# RAG vs CAG Fidelity Evaluation in Healthcare QA

## Overview
This project evaluates the fidelity performance of Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG) in a static question-answering scenario.

The study focuses on healthcare-related queries and compares how different architectures affect answer faithfulness.

## Motivation
Large Language Models (LLMs) often suffer from hallucination issues.  
This project explores whether integrating retrieval (RAG) or caching (CAG) mechanisms can improve answer reliability.

## Tech Stack
- LLM: Llama-3-Chinese-8B-Instruct
- Framework: LangChain
- Evaluation: RAGAS
- Language: Python

## Methodology
- Designed 10 healthcare-related queries
- Built two pipelines:
  - RAG (retrieval-based)
  - CAG (cache-based)
- Evaluated responses using RAGAS metrics:
  - Faithfulness

## Results
| Method | Faithfulness |
|--------|------------|
| RAG    | 0.8         |
| CAG    | 0.903       |

> CAG shows better performance in static QA scenarios due to reduced retrieval noise.
