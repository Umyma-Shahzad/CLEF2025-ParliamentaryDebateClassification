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
#fine tuning the model
def train_model(model_name, tokenizer, train_dataset, test_dataset, num_labels, output_dir, epochs=3, batch_size=8):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predictions = np.argmax(probs, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average='weighted')
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer

# Evaluating and visualizing results of fine tuned model
def evaluate_model(predictions, labels, class_names):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print("\nClassification Report:\n", report)

    precision = precision_score(labels, predictions, average=None)
    recall = recall_score(labels, predictions, average=None)
    f1 = f1_score(labels, predictions, average=None)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')

    print("\nClass-wise Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"Class {class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")
    print(f"\nMacro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

def evaluate_and_visualize(trainer, test_dataset, class_names):
    """Evaluate the model and visualize results."""
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predicted_labels = np.argmax(probs, axis=-1)
    true_labels = predictions.label_ids
    evaluate_model(predicted_labels, true_labels, class_names)


def main(train_folder, country, augmentation_ratio=1.5):
    tokenizer_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
    
    #fine tuning model for Task 01 using text_en
    print("\nTask 1: Political Ideology Identification using text_en")
    orientation_train_dataset = prepare_dataset(orientation_train, tokenizer, text_column='text_en')
    orientation_test_dataset = prepare_dataset(orientation_test, tokenizer, text_column='text_en')
    orientation_trainer = train_model(
        model_name=tokenizer_name,
        tokenizer=tokenizer,
        train_dataset=orientation_train_dataset,
        test_dataset=orientation_test_dataset,
        num_labels=2,
        output_dir="task1_output"
    )
    evaluate_and_visualize(orientation_trainer, orientation_test_dataset, ["Left", "Right"])

    #fine tuning model for Task 02 using text
    print("\nTask 2: Power Identification using text")
    power_train_dataset = prepare_dataset(power_train, tokenizer, text_column='text')
    power_test_dataset = prepare_dataset(power_test, tokenizer, text_column='text')
    power_trainer = train_model(
        model_name=tokenizer_name,
        tokenizer=tokenizer,
        train_dataset=power_train_dataset,
        test_dataset=power_test_dataset,
        num_labels=2,
        output_dir="task2_output"
    )
    evaluate_and_visualize(power_trainer, power_test_dataset, ["Coalition", "Opposition"])


if __name__ == "__main__":
    main(train_folder="train", country="fr", augmentation_ratio=1.5)
