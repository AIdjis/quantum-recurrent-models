# Hybrid Classical–Quantum Recurrent Models for Sarcasm Detection

This project implements and evaluates hybrid classical–quantum recurrent neural network models for sarcasm detection. Classical recurrent layers process the input sequence, while a parameterized quantum circuit is used as a trainable component of the model.

## Overview

The experiments compare hybrid quantum recurrent models with classical baselines on a sarcasm detection dataset. The models are evaluated as binary text classifiers using standard classification metrics.

## Workflow

1. Load and inspect the sarcasm detection dataset.
2. Clean and tokenize the text.
3. Convert tokens to numerical sequences and create training, validation, and test splits.
4. Train classical recurrent and hybrid classical–quantum recurrent models.
5. Evaluate the models on the held-out test set.
6. Compare results and analyze model performance.

## Models

- **Classical baseline:** recurrent text-classification model.
- **Hybrid model:** classical recurrent layers combined with a variational quantum circuit.

The quantum circuit is used through a differentiable interface so that its parameters can be optimized together with the classical network.

## Evaluation

Performance should be reported using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

When comparing models, use the same preprocessing, data splits, training budget, and evaluation procedure. Because quantum simulation can be computationally expensive, results should also include the simulator or quantum backend and relevant circuit settings.

## Reproducibility

Record the following for each experiment:

- Dataset version and split sizes
- Random seed
- Tokenizer and vocabulary settings
- Model architecture and hyperparameters
- Number of qubits and circuit depth
- Optimizer, learning rate, and batch size
- Number of training epochs
- Classical simulator or quantum hardware backend
