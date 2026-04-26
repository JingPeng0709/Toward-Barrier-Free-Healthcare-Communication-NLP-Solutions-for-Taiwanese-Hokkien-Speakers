# Hokkien Translation Module
## Task Definition
This module focuses on translating Mandarin Chinese into Hokkien.
## Motivation
Hokkien is widely used in daily communication, especially among elderly populations. However, most medical information is provided in Mandarin, creating a communication gap.
This module aims to improve accessibility by enabling accurate translation into Hokkien.
## Dataset
- Small-scale curated dataset
- Manually aligned Mandarin–Hokkien pairs
## Evaluation
- BLEU score (for n-gram based lexical similarity)
- Chrf score (character n-gram F-score, suitable for Hokkien romanization)
## Example
| Mandarin | Hokkien |
|----------|----------------|
| 我每天騎腳踏車運動 | guá ta̍ k-kang khiâ kha-ta̍ h-tshia ūn-tōng |
| 可以吃止痛藥嗎 | É-sái chia tsí-thiàng-ioh--bô |
## Future Work
- Expand dataset size
- Improve domain-specific translation (medical terminology)

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
