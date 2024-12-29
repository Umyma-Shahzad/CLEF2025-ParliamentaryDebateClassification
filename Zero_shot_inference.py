import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer,TrainingArguments
from datasets import Dataset
import torch
import re
import nltk
from nltk.corpus import wordnet
import random
nltk.download('wordnet')
nltk.download('omw-1.4')
from transformers import pipeline
import tensorflow as tf
from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset


# Step 1: Loading and preprocessing the dataset
def preprocess_data(data, placeholder="Translation missing"):
    """
    Preprocessing the dataset by cleaning text and handling missing translations.
    """
    def clean_text(text):
        if not isinstance(text, str):
            return text
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    data['text'] = data['text'].apply(clean_text)
    data['text_en'] = data['text_en'].apply(clean_text)
    data['text_en'] = data['text_en'].fillna(placeholder)
    data['text'] = data['text'].fillna(placeholder)
    if 'label' in data.columns:
        data = data.dropna(subset=['label'])
    return data

def load_datatrain(folder_path, task, country):
    file_path = os.path.join(folder_path, task, f"{task}-{country}-train.tsv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    data = pd.read_csv(file_path, sep='\t')
    data = preprocess_data(data)  # Preprocessing the data
    return data

# Step 2: Performing Exploratory Data Analysis
def explore_data(data, name="Dataset"):
    print(f"\nExploratory Data Analysis: {name}")
    print("Dataset Statistics:")
    print(data.describe())

    # Label distribution
    if 'label' in data.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=data['label'])
        plt.title(f"Label Distribution: {name}")
        plt.show()

    # Text length distribution
    data['text_length'] = data['text'].str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(data['text_length'], bins=30, kde=True)
    plt.title(f"Text Length Distribution: {name}")
    plt.show()

# Step 3: Splitting the data (90%-10%)
def split_data(data, stratify_column='label', test_size=0.1):
    """Splitting data into train and test sets while maintaining class proportions."""
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data[stratify_column], random_state=42
    )
    return train_data, test_data

# Step 4: Tokenizing the data
def tokenize_data(data, tokenizer, text_column='text'):
    """Tokenizing text data using the specified tokenizer."""
    return tokenizer(list(data[text_column]), padding=True, truncation=True, max_length=512, return_tensors='pt')

# Step 5: Preparing the dataset
def prepare_dataset(data, tokenizer, text_column, label_column='label'):
    """Preparing dataset for training."""
    tokenized_data = tokenize_data(data, tokenizer, text_column)
    dataset_dict = {
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask']
    }
    if label_column in data.columns:
        dataset_dict['labels'] = torch.tensor(data[label_column].values, dtype=torch.long)
    return Dataset.from_dict(dataset_dict)

# Step 6: Applying augmentation techniques to handle class imbalance

# Synonym Replacement
def synonym_replacement(text, max_synonyms=2):
    """
    Replacing words in the text with their synonyms.
    """
    words = text.split()
    new_words = words.copy()
    for i, word in enumerate(words):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        synonyms.discard(word)  # Avoid replacing with the same word
        if synonyms:
            new_word = random.choice(list(synonyms))
            new_words[i] = new_word
            max_synonyms -= 1
            if max_synonyms <= 0:
                break
    return ' '.join(new_words)

# Random Insertion
def random_insertion(text, n=2):
    """
    Inserting random words from the text into the sentence.
    """
    words = text.split()
    for _ in range(n):
        random_word = random.choice(words)
        random_position = random.randint(0, len(words)-1)
        words.insert(random_position, random_word)
    return ' '.join(words)

def augment_data(data, text_column, label_column='label', augmentation_ratio=1.5):
    """
    Augmenting the data to handle class imbalance by applying text augmentation on the minority class.
    """
    augmented_data = data.copy()

    class_counts = data[label_column].value_counts()

    min_class = class_counts.idxmin()
    min_class_count = class_counts[min_class]

    target_class_count = int(min_class_count * augmentation_ratio)

    # Augmenting the minority class
    minority_class_data = data[data[label_column] == min_class]
    minority_class_data = minority_class_data.reset_index(drop=True)

    augmented_texts = []

    while len(augmented_texts) < target_class_count:
        text = random.choice(minority_class_data[text_column].tolist())
        augmented_text_1 = synonym_replacement(text)
        augmented_text_2 = random_insertion(text)

        augmented_texts.append(augmented_text_1)
        augmented_texts.append(augmented_text_2)

    augmented_df = pd.DataFrame({text_column: augmented_texts, label_column: [min_class] * len(augmented_texts)})
    data = pd.concat([data, augmented_df], ignore_index=True)

    return data

#performing zero-shot inference
def zero_shot_inference_with_dataset(model_name, test_dataset, text_column, task_name):
    print(f"Starting zero-shot inference for {task_name} using {text_column}...")
    classifier = pipeline("zero-shot-classification", model=model_name, device=0)

    if task_name == "Task 01 (Political Ideology)":
        candidate_labels = ["left-wing", "right-wing"]
    elif task_name == "Task 02 (Political Orientation)":
        candidate_labels = ["governing party", "opposition party"]

    predictions = []
    for i, result in enumerate(classifier(KeyDataset(test_dataset, text_column), candidate_labels=candidate_labels, batch_size=8)):
        predicted_label = result['labels'][0]
        predictions.append(candidate_labels.index(predicted_label))
        if i % 100 == 0:
            print(f"Processed {i} samples...")

    labels = test_dataset['label']
    print(f"Completed zero-shot inference for {task_name} using {text_column}.")
    return labels, predictions

#visualizing results of zero-shot inference
def generate_report_and_visualizations(labels, predictions, task_name, text_type):
    report = classification_report(labels, predictions, target_names=["0", "1"])
    print(f"Classification Report for {task_name} using {text_type}:\n", report)
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    if task_name == "Task 01 (Political Ideology)":
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["left", "right"], yticklabels=["left", "right"])
    else:
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["governing party", "opposition party"], yticklabels=["governing party", "opposition party"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {task_name} using {text_type}')
    plt.show()

    metrics = classification_report(labels, predictions, output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=metrics_df.index[:-3], y=metrics_df['precision'][:-3], color='b', label='Precision')
    sns.barplot(x=metrics_df.index[:-3], y=metrics_df['recall'][:-3], color='r', label='Recall', alpha=0.5)
    sns.barplot(x=metrics_df.index[:-3], y=metrics_df['f1-score'][:-3], color='g', label='F1-Score', alpha=0.3)

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title(f'Precision, Recall, and F1-Score for {task_name} using {text_type}')
    plt.legend()
    plt.show()

def main(train_folder, country, augmentation_ratio=1.5):

    print("Starting main function...")
    orientation_task = "orientation"
    power_task = "power"

    orientation_data = load_datatrain(train_folder, orientation_task, country)
    power_data = load_datatrain(train_folder, power_task, country)

    print("\nBefore Augmentation")
    explore_data(orientation_data, "Orientation")
    explore_data(power_data, "Power")

    orientation_data = augment_data(orientation_data, text_column='text_en', label_column='label', augmentation_ratio=1.5)
    power_data = augment_data(power_data, text_column='text', label_column='label', augmentation_ratio=1.5)

    print("\nAfter Augmentation")
    explore_data(orientation_data, "Orientation")
    explore_data(power_data, "Power")

    orientation_train, orientation_test = split_data(orientation_data)
    power_train, power_test = split_data(power_data)

    orientation_test = orientation_test.dropna(subset=['text'])
    orientation_test = orientation_test.dropna(subset=['text_en'])
    power_test = power_test.dropna(subset=['text'])
    power_test = power_test.dropna(subset=['text_en'])

    orientation_test_dataset = Dataset.from_pandas(orientation_test)
    power_test_dataset = Dataset.from_pandas(power_test)

    # Zero-shot inference for Task 01 using 'text'
    labels_01_text, predictions_01_text = zero_shot_inference_with_dataset("facebook/bart-large-mnli", orientation_test_dataset, 'text', "Task 01 (Political Ideology)")
    generate_report_and_visualizations(labels_01_text, predictions_01_text, "Task 01 (Political Ideology)", "text")

    # Zero-shot inference for Task 01 using 'text_en'
    labels_01_text_en, predictions_01_text_en = zero_shot_inference_with_dataset("facebook/bart-large-mnli", orientation_test_dataset, 'text_en', "Task 01 (Political Ideology)")
    generate_report_and_visualizations(labels_01_text_en, predictions_01_text_en, "Task 01 (Political Ideology)", "text_en")

    # Zero-shot inference for Task 02 using 'text'
    labels_02_text, predictions_02_text = zero_shot_inference_with_dataset("facebook/bart-large-mnli", power_test_dataset, 'text', "Task 02 (Political Orientation)")
    generate_report_and_visualizations(labels_02_text, predictions_02_text, "Task 02 (Political Orientation)", "text")

    # Zero-shot inference for Task 02 using 'text_en'
    labels_02_text_en, predictions_02_text_en = zero_shot_inference_with_dataset("facebook/bart-large-mnli", power_test_dataset, 'text_en', "Task 02 (Political Orientation)")
    generate_report_and_visualizations(labels_02_text_en, predictions_02_text_en, "Task 02 (Political Orientation)", "text_en")

if __name__ == "__main__":
    main(train_folder="train", country="fr", augmentation_ratio=1.5)
