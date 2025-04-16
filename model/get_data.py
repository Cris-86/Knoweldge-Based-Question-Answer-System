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
                tokenized_question = preprocess_document(data['question'])
                datasets.append({
                    'question': data['question'],
                    'tokenized_question': tokenized_question,
                    'answer': data['answer'],
                    'document_id': data['document_id']
                })
    return datasets, len(datasets)

def get_pyserini_path(file_path):
    return os.path.join(os.path.dirname(file_path), 'pyserini_' + os.path.basename(file_path))

def trans2pyserini(file_path='./data/documents.jsonl'):
    pyserini_path = get_pyserini_path(file_path)
    if not os.path.exists(pyserini_path):
        with open(file_path, 'r', encoding='utf-8') as f_in, open(pyserini_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm.tqdm(f_in, desc="Converting to Pyserini format"):
                if line.strip():
                    doc = json.loads(line)
                    dpr_format = {
                        "id": str(doc["document_id"]),
                        "contents": doc["document_text"]
                    }
                    f_out.write(json.dumps(dpr_format) + "\n")

def build_dpr_index(file_path='./data/documents.jsonl'):
    processed_path = trans2pyserini(file_path)
    
    encode_cmd = f"""
    python -m pyserini.encode \
      input   --corpus {processed_path} \
              --fields text \
      output  --embeddings dpr_index \
      encoder --encoder facebook/dpr-ctx_encoder-multiset-base \
              --fields text \
              --batch-size 32 \
              --device cpu
    """
    
    index_cmd = """
    python -m pyserini.index.faiss \
      --input dpr_index \
      --output dpr_faiss_index \
      --hnsw
    """
    
    print("Running encoding command:")
    os.system(encode_cmd)
    print("\nBuilding FAISS index:")
    os.system(index_cmd)
