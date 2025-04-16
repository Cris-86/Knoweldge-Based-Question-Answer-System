from gensim.models import KeyedVectors
import get_data
import os
import numpy as np
import tqdm
from collections import Counter
import math
import pickle
from scipy.spatial.distance import cdist

class Word2Vec_KBQA:
    def __init__(self, model_path='./model/Word2Vec.bin', file_path='./data/documents.jsonl', top_k=5, refine_model=False, use_wandb=False):
        self.model_path = model_path
        self.file_path = file_path
        self.top_k = top_k
        self.refine_model = refine_model
        self.use_wandb = use_wandb

        name = self.get_file_name(file_path)
        if name == 'documents':
            self.documents, self.tokenized_docs, self.documentsLen = get_data.load_documents(file_path)
        else:
            data, len_data = get_data.load_datasets(file_path)
            self.documents = data['question']
            self.tokenized_docs = data['tokenized_question']
        self.model = self.load_word2vec_model(model_path)
        self.vector_size = self.model.vector_size
        self.calculate_idf()
        self.doc_vectors = self.document_to_vector()
        self.precompute_common_word_vectors()
    
    def load_word2vec_model(self, model_path):
        '''
        cache_path = model_path + '.cache'
        if os.path.exists(cache_path):
            print("Loading cached word2vec model...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        print("Loading and optimizing word2vec model...")
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        all_tokens = set()
        for doc in self.tokenized_docs:
            all_tokens.update(doc)
        common_query_words = {"what", "who", "where", "when", "why", "how", "which", 
                             "is", "are", "was", "were", "will", "would", "can", "could"}
        all_tokens.update(common_query_words)
        filtered_model = KeyedVectors(vector_size=model.vector_size)
        for token in all_tokens:
            if token in model:
                filtered_model.add_vector(token, model[token])
        with open(cache_path, 'wb') as f:
            pickle.dump(filtered_model, f)
        return filtered_model
        '''

        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        return model

    def precompute_common_word_vectors(self):
        word_counts = Counter()
        for doc in self.tokenized_docs:
            word_counts.update(doc)

        common_words = [word for word, _ in word_counts.most_common(1000)]

        common_query_words = ["what", "who", "where", "when", "why", "how", "which"]
        common_words.extend(common_query_words)

        self.common_word_vectors = {}
        for word in common_words:
            if word in self.model:
                self.common_word_vectors[word] = self.model[word]

        
    def get_file_name(self, file_path):
        file_name_with_extension = os.path.basename(file_path)
        documents = os.path.splitext(file_name_with_extension)[0]
        return documents
    
    def calculate_idf(self):
        idf_cache_path = os.path.join(os.path.dirname(self.file_path), 'idf_values.pkl')
        
        if os.path.exists(idf_cache_path):
            with open(idf_cache_path, 'rb') as f:
                self.idf = pickle.load(f)
        else:
            self.term_doc_freq = Counter()
            self.doc_count = len(self.tokenized_docs)
            
            for doc in tqdm.tqdm(self.tokenized_docs, desc="Calculating IDF values"):
                terms_in_doc = set(doc)  
                for term in terms_in_doc:
                    self.term_doc_freq[term] += 1
            
            self.idf = {}
            for term, doc_freq in self.term_doc_freq.items():
                self.idf[term] = math.log(self.doc_count / (1 + doc_freq))
            
            with open(idf_cache_path, 'wb') as f:
                pickle.dump(self.idf, f)
    
    def document_to_vector(self):
        savepath = os.path.join(os.path.dirname(self.file_path), 'document_vectors_idf_weighted.npy')
        if os.path.exists(savepath):
            return np.load(savepath)
        else:
            print("Converting documents to vectors (this may take a while but will be cached)...")
            batch_size = 1000
            total_docs = len(self.tokenized_docs)
            vectors = np.zeros((total_docs, self.vector_size))
            
            for i in tqdm.tqdm(range(0, total_docs, batch_size), desc="Processing document batches"):
                end_idx = min(i + batch_size, total_docs)
                for j in range(i, end_idx):
                    vectors[j] = self.text_to_vector(self.tokenized_docs[j])
            
            np.save(savepath, vectors)
            return vectors
    
    def text_to_vector(self, tokens):
        vector = np.zeros(self.vector_size)
        total_weight = 0

        for token in tokens:
            if token in self.common_word_vectors:
                word_vector = self.common_word_vectors[token]
            elif token in self.model:
                word_vector = self.model[token]
            else:
                continue
            idf_weight = self.idf.get(token, 1.0)
            vector += word_vector * idf_weight
            total_weight += idf_weight
        
        if total_weight > 0:
            vector /= total_weight
            
        return vector

    
    def query_to_vector(self, query_tokens):
        return self.text_to_vector(query_tokens)

    
    def batch_cosine_similarity(self, query_vector, doc_vectors):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        similarities = 1 - cdist(query_vector, doc_vectors, 'cosine')[0]
        return similarities

    
    def cosine_similarity(self, v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        epsilon = 1e-10 
        return dot / (max(norm1 * norm2, epsilon))
    
    def retrieve_single_question(self, question):
        question_tokens = get_data.preprocess_document(question)
        query_vector = self.query_to_vector(question_tokens)
        similarities = self.batch_cosine_similarity(query_vector, self.doc_vectors)
        top_indices = np.argsort(-similarities)[:self.top_k]
        top_k_indices = top_indices.tolist()
        top_k_docs = [self.tokenized_docs[idx] for idx in top_k_indices]      
        return top_k_indices, top_k_docs
    
    def create_focused_vector(self, doc_tokens, query_tokens):
        """Create a vector focused on terms in document that are most relevant to query"""
        # Find terms that appear in both document and query
        common_terms = set(doc_tokens).intersection(set(query_tokens))
        
        # If no common terms, find the most similar terms using word vectors
        if len(common_terms) == 0:
            focused_tokens = self.find_similar_terms(doc_tokens, query_tokens)
        else:
            focused_tokens = list(common_terms)
            
        # Add some context around common terms from the document
        context_tokens = []
        for i, token in enumerate(doc_tokens):
            if token in focused_tokens:
                # Add surrounding tokens for context
                start = max(0, i-2)
                end = min(len(doc_tokens), i+3)
                context_tokens.extend(doc_tokens[start:end])
        
        # Combine common terms and context
        all_focused_tokens = focused_tokens + context_tokens
        
        # Calculate vector with higher weights for common terms
        vector = np.zeros(self.vector_size)
        total_weight = 0
        
        for token in all_focused_tokens:
            if token in self.model:
                # Higher weight for tokens that appear in both document and query
                weight = 2.0 if token in focused_tokens else 1.0
                vector += self.model[token] * weight
                total_weight += weight
        
        if total_weight > 0:
            vector /= total_weight
            
        return vector
    
    def find_similar_terms(self, doc_tokens, query_tokens, top_n=5):
        """Find terms in document that are most similar to query terms"""
        similar_terms = []
        
        # Filter tokens that are in the model
        doc_tokens_in_model = [t for t in doc_tokens if t in self.model]
        query_tokens_in_model = [t for t in query_tokens if t in self.model]
        
        if not doc_tokens_in_model or not query_tokens_in_model:
            return doc_tokens[:10] if doc_tokens else []
        
        # For each query token, find most similar document tokens
        for q_token in query_tokens_in_model:
            q_vector = self.model[q_token]
            
            # Calculate similarity with each document token
            token_sims = []
            for d_token in doc_tokens_in_model:
                if d_token in self.model:
                    sim = self.cosine_similarity(q_vector, self.model[d_token])
                    token_sims.append((d_token, sim))
            
            # Add top similar terms
            if token_sims:
                token_sims.sort(key=lambda x: x[1], reverse=True)
                similar_terms.extend([t for t, _ in token_sims[:top_n]])
        
        return list(set(similar_terms))  # Remove duplicates
    
    def retrieve_datasets(self, tokenized_question):
        query_vector = self.query_to_vector(tokenized_question)

        similarities = self.batch_cosine_similarity(query_vector, self.doc_vectors)

        top_indices = np.argsort(-similarities)[:self.top_k]

        top_k_indices = top_indices.tolist()
        top_k_docs = [self.tokenized_docs[idx] for idx in top_k_indices]
        
        return top_k_indices, top_k_docs

    
    def ensure_diversity(self, indices, docs, query):
        """Ensure diversity in retrieved documents"""
        if len(indices) <= 1:
            return indices, docs
            
        # Always keep the top result
        selected_indices = [indices[0]]
        selected_docs = [docs[0]]
        
        # For remaining positions, select diverse documents
        remaining_indices = indices[1:]
        remaining_docs = docs[1:]
        
        query_vector = self.query_to_vector(query)
        
        while len(selected_indices) < self.top_k and remaining_indices:
            # Find the document that is most different from already selected ones
            # but still relevant to the query
            max_diversity_score = -float('inf')
            best_idx = 0
            
            for i, (idx, doc) in enumerate(zip(remaining_indices, remaining_docs)):
                doc_vector = self.doc_vectors[idx]
                
                # Calculate relevance to query
                query_relevance = self.cosine_similarity(query_vector, doc_vector)
                
                # Calculate diversity (average dissimilarity to selected docs)
                diversity = 0
                for selected_idx in selected_indices:
                    selected_vector = self.doc_vectors[selected_idx]
                    # 1 - similarity = dissimilarity
                    diversity += 1 - self.cosine_similarity(doc_vector, selected_vector)
                
                if selected_indices:
                    diversity /= len(selected_indices)
                
                # Balance relevance and diversity (70% relevance, 30% diversity)
                diversity_score = 0.7 * query_relevance + 0.3 * diversity
                
                if diversity_score > max_diversity_score:
                    max_diversity_score = diversity_score
                    best_idx = i
            
            # Add the most diverse document
            selected_indices.append(remaining_indices[best_idx])
            selected_docs.append(remaining_docs[best_idx])
            
            # Remove from remaining
            remaining_indices.pop(best_idx)
            remaining_docs.pop(best_idx)
        
        return selected_indices, selected_docs
