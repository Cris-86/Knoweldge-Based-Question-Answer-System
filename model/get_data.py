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
import pickle

def preprocess_document(text):
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")  

    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\-''\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return tokens

def preprocess_text(text):
    # Remove HTML tags and decode HTML entities
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")  

    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def get_processed_path(file_path):
    return os.path.join(os.path.dirname(file_path), 'processed_' + os.path.basename(file_path))

def load_documents(file_path='./data/documents.jsonl'):
    documents = {}
    tokenized_docs = []
    processed_path = get_processed_path(file_path) 
    
    if not os.path.exists(processed_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc="Loading documents"):
                if line.strip():
                    document = json.loads(line)
                    doc_id = document["document_id"]
                    processed_text = preprocess_document(document['document_text'])
                    documents[doc_id] = processed_text 
                    tokenized_docs.append(processed_text)
        
        with open(processed_path, 'w', encoding='utf-8') as f:
            for doc_id, tokens in documents.items():
                f.write(json.dumps({
                    "document_id": doc_id,
                    "document_text": tokens
                }) + '\n')
    else:
        with open(processed_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc="Loading processed documents"):
                document = json.loads(line)
                doc_id = document["document_id"]
                tokens = document["document_text"]
                documents[doc_id] = tokens
                tokenized_docs.append(tokens)
    
    return documents, tokenized_docs, len(documents)

def load_datasets(file_path='./data/train.jsonl'):
    datasets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc="Loading datasets"):
            if line.strip():
                data = json.loads(line)
                print(data['question'])
                tokenized_question = preprocess_document(data['question'])
                datasets.append({
                    'question': data['question'],
                    'tokenized_question': tokenized_question,
                    'answer': data['answer'],
                    'document_id': data['document_id']
                })
    return datasets, len(datasets)

def get_ColBERT_preprocessed_path(file_path):
    return os.path.join(os.path.dirname(file_path), 'ColBERT_processed_document.pkl')

def process_documents_ColBERT(input_path='./data/documents.jsonl', max_passage_length=512):
    
    cache_path = get_ColBERT_preprocessed_path(input_path)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            return cache_data['doc_ids'], cache_data['doc_texts']

    doc_ids = []
    doc_texts = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc="Processing documents"):
            data = json.loads(line)
            doc_id = data['document_id']
            text = preprocess_text(data['document_text'])
            
            sentences = text.split('. ')
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                sent += '. '
                sent_length = len(sent)
                
                if current_length + sent_length > max_passage_length and current_chunk:
                    chunk = ''.join(current_chunk).strip()
                    if chunk:
                        doc_ids.append(doc_id)
                        doc_texts.append(chunk)
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sent)
                current_length += sent_length
            
            if current_chunk:
                chunk = ''.join(current_chunk).strip()
                if chunk:
                    doc_ids.append(doc_id)
                    doc_texts.append(chunk)
    
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'doc_ids': doc_ids,
            'doc_texts': doc_texts
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return doc_ids, doc_texts