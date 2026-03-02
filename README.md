# Disease Prediction from Clinical Symptoms
This project implements a deep learning pipeline to classify medical conditions based on natural language symptom descriptions. By leveraging ClinicalBERT, the system transforms unstructured patient notes into high-dimensional embeddings to predict the most likely diagnosis across 121 unique disease categories.

## Project Objective
The goal is to bridge the gap between raw clinical text and automated diagnosis. I utilized Pre-trained Language Models (PLMs) specifically trained on medical corpora to understand the semantic nuances of symptoms, such as the difference between "acute pain" and "chronic discomfort."

## Main Python Libraries

*    **PyTorch**: The primary framework for neural network construction and gradient-based optimization.
*    **Transformers (Hugging Face)**: Used to implement ClinicalBERT for domain-specific tokenization and text embedding.
*    **Scikit-Learn**: Utilized for intelligent data splitting and calculating class weights to handle dataset imbalances.
*    **Pandas & NumPy**: Essential for cleaning raw symptom data and performing matrix operations.
*    **Matplotlib**: Used for visualizing disease distributions and training progress.

## Data Preparation & Preprocessing
### 1. Feature Engineering

The raw data initially consisted of one-hot encoded symptom columns. To make this compatible with NLP models, a One-Hot Reversal process was used:

*    Individual symptoms were aggregated into a single, human-readable string (e.g., "fever, cough, fatigue").
*    These strings were stored in a note column, serving as the primary textual input for the model.

### 2. Dataset Refinement

To ensure the model learned from statistically significant patterns, a multi-stage filtering process was applied:

*    Frequency Filtering: Only diseases with high instance counts were retained, ensuring the model had enough examples to generalize.
*    Missing Value Handling: Rows with incomplete data were removed to maintain data integrity.
*    Imbalance Mapping: I analyzed the distribution of the 121 disease labels to prepare for weighted loss strategies.

## Model Evolution & Iterative Improvements
The project followed an iterative development cycle, moving from a basic baseline to a sophisticated classification system.

### Phase 1: Embedding & Baseline
I utilized medicalai/ClinicalBERT to convert symptom notes into 768-dimensional vectors. Initially, a simple linear layer was used as a classifier. While this provided a starting point, it highlighted the need for more complex architectures and better optimization.

### Phase 2: Architectural Enhancements (ComplexClassifier)
The model was upgraded to a multi-layer architecture to better capture non-linear relationships:

*    **Linear Layers**: Expanded to include hidden layers for deeper feature extraction.
*    **ReLU Activation**: Introduced to allow the model to learn complex symptom patterns.
*    **Dropout Regularization**: Implemented to prevent the model from memorizing the training data, improving its performance on unseen cases.

### Phase 3: Advanced Training Strategies

*    **Optimizer Upgrade**: Switched from standard SGD to AdamW, which significantly improved convergence speed and stability.
*    **Data Augmentation (EDA)**: Techniques like synonym replacement, random insertion, and word swapping were used to artificially increase the diversity of the training set.
*    **Weighted Cross-Entropy Loss**: To protect minority classes (rare diseases), I penalized misclassifications of infrequent labels more heavily.
*    **Gradient Accumulation**: This allowed the simulation of larger batch sizes, leading to smoother training and more reliable weight updates.

## Summary of Findings
The experiments demonstrated that model complexity and data handling are equally important. While the pre-trained embeddings provide the "medical knowledge," the custom classifier and augmentation strategies provide the "diagnostic logic." The final model showed a massive improvement over the baseline, successfully identifying patterns across over a hundred different medical conditions.

## Future Work

*    **Fine-tuning ClinicalBERT**: Transition from using fixed embeddings to "unfreezing" the BERT layers. This would allow the model to adapt its internal linguistic understanding specifically to this disease dataset.
*    **Generative Augmentation**: Utilizing models like T5 or GPT to create synthetic clinical notes, further enriching the training data for rare conditions.
*    **Attention Mechanisms**: Integrating self-attention into the classifier to help the model focus on the most "pathognomonic" (clinically significant) symptoms in a note.
*    **Back-Translation**: Improving text diversity by translating symptom descriptions into multiple languages and back to English to capture different ways patients describe their health.


## Data Acquisition & Clinical Models
### Clinical Language Models
The core of the text understanding pipeline is built on [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT), a BERT-based model pre-trained on a large-scale clinical corpus.

*    Model Source: medicalai/ClinicalBERT on Hugging Face
*    Reference:
     *  Wang, G., et al. (2023). Optimized glycemic control of type 2 diabetes with reinforcement learning. Nature Medicine. doi:10.1038/s41591-023-02552-9
     *  Wang, G., et al. (2025). A Generalist Medical Language Model for Disease Diagnosis Assistance. Nature Medicine. doi:10.1038/s41591-024-03416-6

### Primary Dataset
The model was trained on the [Disease-Symptom Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset) by dhivyeshrk from Kaggle.

*    Scale: ~246,000 samples across 773 initial unique diseases.
*    Medical Validation: Symptom clusters and disease naming were cross-referenced with Harvard Health A-Z and NHS Inform Scotland.


