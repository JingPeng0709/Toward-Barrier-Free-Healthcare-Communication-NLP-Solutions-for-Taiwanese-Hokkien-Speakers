# Hokkien Translation Module

## Task Definition
This module focuses on translating Mandarin Chinese into Hokkien(Tailo).

## Motivation
Taiwanese is a low-resource language with limited NLP support.  
Medical communication barriers exist, especially for elderly speakers.

## Challenges
- Lack of large-scale parallel corpus
- Multiple writing systems (POJ, Tailo, Han characters)
- Limited evaluation benchmarks

## Writing System
This project adopts Tailo for standardized Hokkien representation.
  
## Translation model dataset
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

Hokkien Query  
→ Translation Module  
→ RAG Retrieval  
→ LLM Answer Generation  
→ (Optional) Back Translation

## Future Work
- Expand dataset size
- Improve domain-specific translation (medical terminology)

<!--# RAG vs CAG Fidelity Evaluation in Healthcare QA

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

> CAG shows better performance in static QA scenarios due to reduced retrieval noise.-->
