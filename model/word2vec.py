from gensim.models import KeyedVectors
import get_data
import os
import numpy as np
import tqdm
from collections import Counter
import math

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
            # need refine
            data, len_data = get_data.load_datasets(file_path)
            self.documents = data['question']
            self.tokenized_docs = data['tokenized_question']

        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.vector_size = self.model.vector_size
        
        # Calculate IDF values
        self.calculate_idf()
        
        # Generate document vectors with IDF weighting
        self.doc_vectors = self.document_to_vector()
        
    def get_file_name(self, file_path):
        file_name_with_extension = os.path.basename(file_path)
        documents = os.path.splitext(file_name_with_extension)[0]
        return documents
    
    def calculate_idf(self):
        """Calculate IDF values for all words in the corpus"""
        self.term_doc_freq = Counter()
        self.doc_count = len(self.tokenized_docs)
        
        # Count document frequency for each term
        for doc in self.tokenized_docs:
            terms_in_doc = set(doc)  # Count each term only once per document
            for term in terms_in_doc:
                self.term_doc_freq[term] += 1
        
        # Calculate IDF for each term
        self.idf = {}
        for term, doc_freq in self.term_doc_freq.items():
            self.idf[term] = math.log(self.doc_count / (1 + doc_freq))
    
    def document_to_vector(self):
        savepath = os.path.join(os.path.dirname(self.file_path), 'document_vectors_idf_weighted.npy')
        if os.path.exists(savepath):
            return np.load(savepath)
        else:
            vectors = []
            for doc in tqdm.tqdm(self.tokenized_docs, desc="Converting documents to vectors with IDF weighting"):
                doc_vector = self.text_to_vector(doc)
                vectors.append(doc_vector)
            
            vectors = np.array(vectors)
            np.save(savepath, vectors)
            return vectors
    
    def text_to_vector(self, tokens):
        vector = np.zeros(self.vector_size)
        total_weight = 0
        
        # First pass: calculate TF-IDF weighted vector
        for token in tokens:
            if token in self.model:
                # Get IDF weight (default to average if term not in corpus)
                idf_weight = self.idf.get(token, 1.0)
                # Add weighted vector
                vector += self.model[token] * idf_weight
                total_weight += idf_weight
        
        # Normalize by total weight
        if total_weight > 0:
            vector /= total_weight
            
        # Add context-aware weighting - emphasize terms at beginning and end of document
        if len(tokens) > 0:
            position_weighted_vector = np.zeros(self.vector_size)
            weights_sum = 0
            
            # Give more weight to first and last few tokens (topic and conclusion usually)
            important_positions = min(5, len(tokens))
            for i in range(important_positions):
                # Front tokens
                if tokens[i] in self.model:
                    position_weighted_vector += self.model[tokens[i]] * 1.5
                    weights_sum += 1.5
                
                # End tokens
                if i < len(tokens) and tokens[-(i+1)] in self.model:
                    position_weighted_vector += self.model[tokens[-(i+1)]] * 1.2
                    weights_sum += 1.2
            
            if weights_sum > 0:
                position_weighted_vector /= weights_sum
                # Combine with TF-IDF vector (80% TF-IDF, 20% position-weighted)
                vector = 0.8 * vector + 0.2 * position_weighted_vector
                
        return vector
    
    def query_to_vector(self, query_tokens):
        """Convert query to vector with special weighting for question words"""
        '''
        question_words = {"what", "who", "where", "when", "why", "how", "which"}
        
        # Basic vector with TF-IDF weighting
        basic_vector = self.text_to_vector(query_tokens)
        
        # Extract and emphasize key terms (excluding question words)
        key_terms_vector = np.zeros(self.vector_size)
        key_term_count = 0
        
        for token in query_tokens:
            if token not in question_words and token in self.model:
                # Apply higher weight to key terms
                key_terms_vector += self.model[token] * 2.0
                key_term_count += 1
        
        if key_term_count > 0:
            key_terms_vector /= key_term_count
            # Combine vectors (60% key terms, 40% basic vector)
            return 0.6 * key_terms_vector + 0.4 * basic_vector
        '''
        basic_vector = self.text_to_vector(query_tokens)

        return basic_vector
    
    def cosine_similarity(self, v1, v2):
        """Enhanced cosine similarity with epsilon to avoid division by zero"""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        epsilon = 1e-10  # Small constant to avoid division by zero
        return dot / (max(norm1 * norm2, epsilon))
    
    def retrieve_single_question(self, question):
        question = get_data.preprocess_document(question)
        query_vector = self.query_to_vector(question)
        
        # Apply dual-direction cosine similarity
        similarities = []
        for i, doc_vector in enumerate(self.doc_vectors):
            # Forward similarity: how well the query matches the document
            forward_sim = self.cosine_similarity(query_vector, doc_vector)
            
            # Backward similarity: how well key parts of document match the query
            # This helps with partial matches where the document contains the answer
            doc_tokens = self.tokenized_docs[i]
            
            # Get most similar terms from document to query
            if len(doc_tokens) > 0:
                # Create a specialized document vector focusing on terms most relevant to query
                focused_doc_vector = self.create_focused_vector(doc_tokens, question)
                backward_sim = self.cosine_similarity(focused_doc_vector, query_vector)
                
                # Combined similarity score (70% forward, 30% backward)
                combined_sim = 0.7 * forward_sim + 0.3 * backward_sim
                similarities.append((i, combined_sim))
            else:
                similarities.append((i, forward_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in similarities[:self.top_k]]
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

        # Use the same enhanced similarity approach as in retrieve_single_question
        similarities = []
        for i, doc_vector in enumerate(self.doc_vectors):
            # Forward similarity
            forward_sim = self.cosine_similarity(query_vector, doc_vector)
            
            # Backward similarity
            doc_tokens = self.tokenized_docs[i]
            
            if len(doc_tokens) > 0:
                focused_doc_vector = self.create_focused_vector(doc_tokens, tokenized_question)
                backward_sim = self.cosine_similarity(focused_doc_vector, query_vector)
                
                # Combine similarities with weights
                combined_sim = 0.7 * forward_sim + 0.3 * backward_sim
                similarities.append((i, combined_sim))
            else:
                similarities.append((i, forward_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)

        top_k_indices = [idx for idx, _ in similarities[:self.top_k]]
        top_k_docs = [self.tokenized_docs[idx] for idx in top_k_indices]
        '''
        # Return more candidates for better recall
        expanded_k = min(self.top_k + 2, len(similarities))
        top_k_indices = [idx for idx, _ in similarities[:expanded_k]]
        top_k_docs = [self.tokenized_docs[idx] for idx in top_k_indices]
        
        # Post-processing to ensure diversity in results
        if len(top_k_indices) > self.top_k:
            # Keep only top_k most diverse results
            diverse_indices, diverse_docs = self.ensure_diversity(
                top_k_indices, top_k_docs, tokenized_question
            )
            return diverse_indices[:self.top_k], diverse_docs[:self.top_k]
        '''
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