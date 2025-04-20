import get_data
import os
from tqdm import tqdm
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
import pickle

class ColBERT_KBQA:
    def __init__(self, file_path='./data/documents.jsonl', 
                 max_passage_length=512, 
                 nbits=2, 
                 top_k=5, 
                 refine_model=False, 
                 use_wandb=False,
                 enable_sentence_search=False):
        self.max_passage_length = max_passage_length
        self.nbits = nbits
        self.top_k = top_k
        self.file_path = file_path
        self.refine_model = refine_model
        self.use_wandb = use_wandb
        self.enable_sentence_search = enable_sentence_search
        self.sentence_index = None
        
        self.INDEX_NAME = 'rag_colbert_index'
        self.CACHE_DIR = os.path.join(os.path.dirname(file_path), 'colbert_cache')
        self.MAPPING_CACHE = os.path.join(self.CACHE_DIR, 'passage_doc_map.pkl')

        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
        
        self._passage_to_doc_map = None
        self._doc_ids = None
        self._doc_texts = None
        self._searcher = None
    
    def _init_sentence_index(self):
        if self.sentence_index is None and self.enable_sentence_search:
            self.sentence_index = SentenceIndex()

    def retrieve_with_sentence_search(self, question, enable_fine_search=False, sentence_top_k=10):
        doc_indices, _ = self.retrieve_single_question(question)
        if not enable_fine_search or not self.enable_sentence_search:
            return doc_indices, []
        docs = self.get_documents(doc_indices)
        self._init_sentence_index()
        relevant_sentences = self.sentence_index.retrieve_sentences(
            question, docs, top_k=sentence_top_k
        )
        return doc_indices, relevant_sentences
        
    @property
    def doc_ids(self):
        if self._doc_ids is None:
            self._load_documents()
        return self._doc_ids
        
    @property
    def doc_texts(self):
        if self._doc_texts is None:
            self._load_documents()
        return self._doc_texts
    
    @property
    def passage_to_doc_map(self):
        if self._passage_to_doc_map is None:
            self._load_mapping()
        return self._passage_to_doc_map
    
    @property
    def searcher(self):
        if self._searcher is None:
            self._load_searcher()
        return self._searcher
    
    def _load_documents(self):
        doc_cache = os.path.join(self.CACHE_DIR, 'documents.pkl')
        
        if os.path.exists(doc_cache):
            print("Loading documents from cache...")
            with open(doc_cache, 'rb') as f:
                data = pickle.load(f)
                self._doc_ids = data['doc_ids']
                self._doc_texts = data['doc_texts']
        else:
            print("Processing documents...")
            self._doc_ids, self._doc_texts = get_data.process_documents_ColBERT(
                self.file_path, max_passage_length=self.max_passage_length
            )
            with open(doc_cache, 'wb') as f:
                pickle.dump({
                    'doc_ids': self._doc_ids,
                    'doc_texts': self._doc_texts
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_mapping(self):
        if os.path.exists(self.MAPPING_CACHE):
            print("Loading passage-document mapping from cache...")
            with open(self.MAPPING_CACHE, 'rb') as f:
                self._passage_to_doc_map = pickle.load(f)
        else:
            print("Creating passage-document mapping...")
            if self._doc_ids is None:
                self._load_documents()
            
            self._passage_to_doc_map = {i: doc_id for i, doc_id in enumerate(self._doc_ids)}
            with open(self.MAPPING_CACHE, 'wb') as f:
                pickle.dump(self._passage_to_doc_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_searcher(self):
        self.build_indexer()

    def build_indexer(self, checkpoint='colbert-ir/colbertv2.0'):
        if not os.path.exists(self.INDEX_NAME):
            print(f"Building ColBERT index: {self.INDEX_NAME}")
            if self._doc_texts is None:
                self._load_documents()
                
            with Run().context(RunConfig(nranks=1, experiment='rag_system')):
                config = ColBERTConfig(doc_maxlen=self.max_passage_length, nbits=self.nbits, kmeans_niters=4)
                indexer = Indexer(checkpoint=checkpoint, config=config)
                indexer.index(name=self.INDEX_NAME, collection=self.doc_texts, overwrite=True)
        
        print(f"Loading ColBERT searcher from {self.INDEX_NAME}")
        with Run().context(RunConfig(experiment='rag_system')):
            self._searcher = Searcher(
                index=self.INDEX_NAME,
                collection=self.doc_texts
            )
    
    def retrieve_single_question(self, question):
        results = self.searcher.search(question, k=min(self.top_k, 100))
        final_indices = results[0][:self.top_k]
        doc_scores = results[2][:self.top_k]
        return final_indices, doc_scores

    def retrieve_datasets(self, question):  
        # Get results from ColBERT searcher
        results = self.searcher.search(question, k=min(self.top_k*5, 100))
        final_indices = results[0][:self.top_k]
        doc_scores = results[2][:self.top_k]
        return final_indices, doc_scores

    def get_documents(self, doc_indices):
        # Helper method to get full documents by indices
        docs = []
        for idx in doc_indices:
            # Find all passages for this document
            passages = [self.doc_texts[i] for i, doc_id in enumerate(self.doc_ids) if doc_id == idx]
            # Join passages to reconstruct document
            full_doc = " ".join(passages)
            docs.append(full_doc)
        return docs

class SentenceIndex:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.cache_dir = './sentence_index_cache'
        self.sentence_encoder = None
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _load_sentence_encoder(self):
        if self.sentence_encoder is None:
            from sentence_transformers import SentenceTransformer
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve_sentences(self, query, documents, top_k=5):
        self._load_sentence_encoder()
        
        all_sentences = []
        doc_to_sentences = {}
        
        for doc_idx, doc in enumerate(documents):
            sentences = [s.strip() for s in doc.split('.') if s.strip()]
            doc_to_sentences[doc_idx] = sentences
            all_sentences.extend(sentences)
        
        if not all_sentences:
            return []
        query_embedding = self.sentence_encoder.encode(query, convert_to_tensor=True)
        sentence_embeddings = self.sentence_encoder.encode(all_sentences, convert_to_tensor=True)
        from torch.nn import functional as F
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), sentence_embeddings)
        top_indices = similarities.argsort(descending=True)[:top_k].tolist()

        return [all_sentences[idx] for idx in top_indices]