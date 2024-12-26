# Sentiment_Analysis_of_Amazon_Reviews

This repository contains the implementation of a sentiment analysis using advanced Natural Language Processing (NLP) techniques, specifically **GPT-2** and **BERT-BiGRU**, to classify Amazon product reviews into positive or negative sentiments.

## Project Overview

- **Objective**: To classify customer sentiments from Amazon product reviews using state-of-the-art transformer-based NLP models.
- **Dataset**: Amazon Reviews dataset (571.54M reviews from 1996–2023), with a balanced sample of 10,000 reviews used for training and evaluation.
- **Models Used**:
  - **GPT-2**: Fine-tuned for binary sentiment classification.
  - **BERT-BiGRU Hybrid**: Combines BERT embeddings with BiGRU layers and attention mechanisms.

## Features

1. **Data Preprocessing**:
   - Text normalization (removal of URLs, punctuation, and extra whitespaces).
   - Sentiment labeling: Ratings ≥ 4 as "Positive" and ≤ 2 as "Negative".
   - Balanced sampling for robust training.

2. **Model Architectures**:
   - **GPT-2**:
     - Pre-trained GPT-2 fine-tuned for sentiment classification.
     - Self-attention mechanisms for deep contextual understanding.
   - **BERT-BiGRU Hybrid**:
     - DistilBERT embeddings combined with Bidirectional GRU layers.
     - Attention mechanisms to focus on the most relevant text features.

3. **Performance Metrics**:
   - Accuracy, Recall, F1-score.
   - Visualized training loss and accuracy trends over epochs.

4. **Results**:
   - GPT-2 achieved higher accuracy (~95%) compared to BERT (~89%).


