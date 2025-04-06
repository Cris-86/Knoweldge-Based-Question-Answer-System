import json
import jsonlines
import tqdm

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

import os

import re
from bs4 import BeautifulSoup
import html

def preprocess_document(text):
    """
    Preprocess the input document.
    
    Args:
        text (str): The input text to preprocess.
        
    Returns:
        list: A list of preprocessed tokens.
    """
    # Remove HTML tags and decode HTML entities
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")  

    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize the text
    tokens = word_tokenize(text.lower())

    # remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return tokens

def preprocess_text(text):
    """
    Preprocess the input document.
    
    Args:
        text (str): The input text to preprocess.
        
    Returns:
        list: A list of preprocessed tokens.
    """
    # Remove HTML tags and decode HTML entities
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")  

    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_documents(file_path='./data/documents.jsonl'):
    """
    Load documents from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing documents.
        
    Returns:
        list: A list of documents loaded from the file.
        int: The number of documents loaded.
    """
    documents = {}
    tokenized_docs = []
    if not os.path.exists(os.path.join(os.path.dirname(file_path), 'processed_documents.jsonl')):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc="Loading documents"):
                if line.strip():
                    document = json.loads(line)
                    doc_id = document["document_id"]
                    # Preprocess the document text
                    document['document_text'] = preprocess_document(document['document_text'])
                    documents[doc_id] = document['document_text']
                    tokenized_docs.append(tokens)
        save_path = os.path.join(os.path.dirname(file_path), 'processed_documents.jsonl')
        with open(save_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        print(f"Processed documents saved to {save_path}")
    else:
        save_path = os.path.join(os.path.dirname(file_path), 'processed_documents.jsonl')
        with jsonlines.open(save_path) as reader:
            for doc in tqdm.tqdm(reader, desc="Loading processed documents"):
                doc_id = doc["document_id"]
                tokens = doc["document_text"]
                documents[doc_id] = tokens
                tokenized_docs.append(tokens)
    return documents, tokenized_docs, len(documents)

def load_datasets(file_path='./data/train.jsonl'):
    """
    Load datasets from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing datasets.
        
    Returns:
        list: A list of datasets loaded from the file.
        int: The number of datasets loaded.
    """
    datasets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc="Loading datasets"):
            if line.strip():
                data = json.loads(line)
                question = preprocess_text(data['question'])
                datasets.append({
                    'question': question,
                    'answer': data['answer'],
                    'document_id': data['document_id']
                })
    return datasets, len(datasets)