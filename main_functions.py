import json
import stanza
import nltk
import logging
import spacy
import re
import sympy
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


# Load AraBERT model and tokenizer
arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Disable Stanza logging (set to ERROR level)
logging.getLogger('stanza').setLevel(logging.ERROR)

# Download English model
nlp_en = spacy.load("en_core_web_lg")

# Download Arabic model for stanza silently
stanza.download('ar', verbose=False)
nlp_ar = stanza.Pipeline('ar', processors='tokenize,lemma', verbose=False)


def is_arabic(text):
    """Check if the input text contains Arabic characters."""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))


def load_json_file(file_path, file_description):
    """
    Load a JSON file and return its contents. Validate structure and return an empty intents dictionary
    if loading fails.

    Args:
        file_path (str): Path to the JSON file.
        file_description (str): Description of the file for error messages (e.g., "English training data").

    Returns:
        dict: Loaded JSON data or {'intents': []} if loading fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if 'intents' not in data:
            print(f"Error: {file_path} does not contain 'intents' key! Using default data.")
            return {'intents': []}
        print(f"{file_description} loaded successfully, Found {len(data)} valid intents.")
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found! Using default data.")
        return {'intents': []}
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file! Using default data.")
        return {'intents': []}


def preprocess(text, language='en'):
    """
    Preprocess text by tokenizing and lemmatizing based on language.

    Args:
        text (str): Input text to preprocess.
        language (str): Language of the text ('en' or 'ar').

    Returns:
        list: List of lemmatized tokens.
    """
    try:
        if not isinstance(text, str):
            raise ValueError(f"Input to preprocess must be a string, got: {text}")
        if not text.strip():
            print(f"Warning: Empty text input in preprocess, Language: {language}")
            return []
        if language == 'en':
            doc = nlp_en(text.lower())
            tokens = [token.lemma_ for token in doc]
        else:
            doc = nlp_ar(text)
            tokens = [word.lemma for sent in doc.sentences for word in sent.words]
        return tokens
    except Exception as e:
        print(f"Error in preprocess ({language}): {e}")
        return []


def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in tokenized_sentence:
            bag[idx] = 1
    return bag


# Function to get AraBERT embeddings
def get_arabert_embedding(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = arabert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = arabert_model(**inputs)
    # Use [CLS] token embedding as PyTorch tensor
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().to(device)
    return embedding


# Pad or truncate features to match input_size
def pad_or_truncate(features, target_size):
    if len(features) < target_size:
        return np.pad(features, (0, target_size - len(features)), 'constant')
    else:
        return features[:target_size]


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        return out


class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def evaluate_math_expression(expression):
    try:
        # Convert words to operators
        expression = expression.lower()
        if is_arabic(expression):
            expression = expression.lower()
            expression = expression.replace("زائد", "+")
            expression = expression.replace("ناقص", "-")
            expression = expression.replace("مقسوم علي", "/")
            expression = expression.replace("مضروب في", "*")
            expression = expression.replace("اس", "**")
            expression = expression.replace("تربيع", "**2")
            expression = expression.replace("علي", "/")
            expression = expression.replace("^", "**")
        else:
            expression = expression.replace("plus", "+")
            expression = expression.replace("minus", "-")
            expression = expression.replace("times", "*")
            expression = expression.replace("divided by", "/")
            expression = expression.replace("divide by", "/")
            expression = expression.replace("multiply by", "*")
            expression = expression.replace("multiplied by", "*")
            expression = expression.replace("to the power", "**")
            expression = expression.replace("squared", "**2")
            expression = expression.replace("over", "/")
            expression = expression.replace("^", "**")
        # Extract numbers and operations only
        expression = "".join([char for char in expression if char.isdigit() or char in "+-*/(). "])
        # Remove extra spaces
        expression = " ".join(expression.split())
        result = sympy.sympify(expression)  # Calculation of the output
        return float(result)  # Convert it to a floating
    except Exception as e:
        return f"Sorry, I couldn't solve this. Error: {e}"
