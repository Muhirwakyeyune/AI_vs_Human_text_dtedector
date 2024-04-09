import torch
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load training data
train_data = pd.read_csv("/Users/salomonmuhirwa/Downloads/llm-detect-ai-generated-text(1)/train_essays.csv")

# Drop rows with missing values
train_data.dropna(inplace=True)

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define stopwords for NLTK tokenizer
stop_words = set(stopwords.words('english'))

# Initialize NLTK tokenizer
tokenizer_nltk = RegexpTokenizer(r'\w+')

# Function to process a single chunk of text
def process_chunk(text_chunk, max_chunk_length):
    # Tokenize using NLTK tokenizer
    tokenized_text = tokenizer_nltk.tokenize(text_chunk.lower())
    # Remove stopwords
    tokenized_text = [word for word in tokenized_text if word not in stop_words]
    # Truncate or pad the sequence to match max_chunk_length
    tokenized_text = tokenized_text[:max_chunk_length] + ['<pad>'] * (max_chunk_length - len(tokenized_text))
    # Convert tokens to token IDs using GPT-2 tokenizer
    inputs = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Convert token IDs to tensor
    inputs = torch.tensor(inputs).unsqueeze(0)
    return inputs

# Function to chunk the text and process each chunk
def process_text(text):
    max_chunk_length = 128  # Adjust this value as needed
    text_chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    outputs = []
    for chunk in text_chunks:
        chunk_outputs = process_chunk(chunk, max_chunk_length)
        outputs.append(chunk_outputs)
    return outputs

# Process the data in batches
outputs_list = []
for index, row in train_data.iterrows():
    text = row['text']
    batch_outputs = process_text(text)
    outputs_list.extend(batch_outputs)

# Combine outputs from all batches
all_outputs = torch.cat(outputs_list, dim=0)

# Define target vector
y = train_data['generated'].values

# Define cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# List to store cross-validation scores
cv_scores = []

# List to store predicted labels and true labels
all_predicted_labels = []
all_true_labels = []

# Define parameters for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform stratified k-fold cross-validation with Random Forest
for train_index, test_index in skf.split(all_outputs, y):
    X_train, X_test = all_outputs[train_index], all_outputs[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best Random Forest model from grid search
    best_rf = grid_search.best_estimator_
    
    # Model evaluation
    y_pred_test = best_rf.predict(X_test)
    score = accuracy_score(y_test, y_pred_test)
    cv_scores.append(score)
    
    # Append predicted labels and true labels
    all_predicted_labels.extend(y_pred_test)
    all_true_labels.extend(y_test)

# Display cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))

# Generate confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Student', 'LLM'], yticklabels=['Student', 'LLM'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the best Random Forest model
rf_model_path = "/Users/salomonmuhirwa/Downloads/llm-detect-ai-generated-text(1)/rf_model.pkl"
torch.save(best_rf, rf_model_path)
