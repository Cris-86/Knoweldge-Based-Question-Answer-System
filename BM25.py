from rank_bm25 import BM25Okapi
import get_data
import os
import json
from sklearn.model_selection import ParameterGrid
import numpy as np
import tqdm
import wandb

class BM25_KBQA:
    def __init__(self, file_path='./data/documents.jsonl', top_k=5, refine_model=False, use_wandb=False):
        self.top_k = top_k
        self.knowledge_base = []
        name = self.get_file_name(file_path)
        if name == 'documents':
            self.documents, self.tokenized_docs, self.documentsLen = get_data.load_documents(file_path)
        else:
            # need refine
            data, len_data = get_data.load_datasets(file_path)
            self.documents = data['question']
            self.tokenized_docs = data['tokenized_question']
        self.refine_model = refine_model
        self.use_wandb = use_wandb
        
        if self.refine_model:
            if self.use_wandb:
                self.get_wandb_api_key()
                wandb.login(key=self.wandb_api_key) 
                wandb.init(project="NLP_assignment", 
                        config={
                                "top_k": top_k,
                                "file_path": file_path,
                                "model": "BM25",
                                "task": "document retrieval",
                                "description": "BM25 model for document retrieval"
                        }) 
            self.refineModel()
        else:
            self.bm25 = BM25Okapi(self.tokenized_docs, k1=2.8, b=0.6)

    
    def get_file_name(self, file_path):
        file_name_with_extension = os.path.basename(file_path)
        documents = os.path.splitext(file_name_with_extension)[0]
        return documents

    def get_wandb_api_key(self, file_path='wandbKey'):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.wandb_api_key = f.read().strip()
        else:
            raise FileNotFoundError(f"Credentials file '{file_path}' not found.")

    def retrieve_single_question(self, tokenized_question):
        expanded_query = self.expand_query(tokenized_question)
        original_scores = self.bm25.get_scores(tokenized_question)
        expanded_scores = self.bm25.get_scores(expanded_query)
        combined_scores = [0.7 * original_scores[i] + 0.3 * expanded_scores[i] for i in range(len(original_scores))]
        initial_k = min(self.top_k + 3, len(combined_scores))
        top_indices = sorted(range(len(combined_scores)), key=lambda i: -combined_scores[i])[:initial_k]
        final_indices = self.post_process_results(top_indices, tokenized_question)[:self.top_k]
        # best_matches = [self.documents[i] for i in final_indices]
        return final_indices, combined_scores

    def expand_query(self, tokenized_question):
        expanded_query = tokenized_question.copy()
        for token in tokenized_question:
            if token == "when":
                expanded_query.extend(["date", "time", "year"])
            elif token == "where":
                expanded_query.extend(["location", "place", "country"])
            elif token == "who":
                expanded_query.extend(["person", "name", "individual"])
            elif token == "how":
                expanded_query.extend(["method", "way", "means"])     
        return expanded_query
    
    def post_process_results(self, indices, query):
        return indices[:self.top_k]

    
    def calculate_document_difference(self, doc_idx, selected_indices):
        doc_terms = set(self.tokenized_docs[doc_idx])
        avg_difference = 0
        
        for sel_idx in selected_indices:
            sel_terms = set(self.tokenized_docs[sel_idx])
            if len(doc_terms.union(sel_terms)) > 0:
                similarity = len(doc_terms.intersection(sel_terms)) / len(doc_terms.union(sel_terms))
                avg_difference += (1 - similarity)
            
        return avg_difference / len(selected_indices) if selected_indices else 0

    def retrieve_datasets(self, tokenized_question):
        expanded_query = self.expand_query(tokenized_question)
        original_scores = self.bm25.get_scores(tokenized_question)
        expanded_scores = self.bm25.get_scores(expanded_query)
        combined_scores = [0.7 * original_scores[i] + 0.3 * expanded_scores[i] for i in range(len(original_scores))]
        initial_k = min(self.top_k + 3, len(combined_scores))
        top_indices = sorted(range(len(combined_scores)), key=lambda i: -combined_scores[i])[:initial_k]
        final_indices = self.post_process_results(top_indices, tokenized_question)[:self.top_k]
        # best_matches = [self.documents[i] for i in final_indices]
        return final_indices, combined_scores
    
    def refineModel(self, train_path = "./data/val.jsonl"):
        train_data, len_train_data = get_data.load_datasets(train_path)
        param_grid = {
            "k1": np.arange(1.0, 2.5, 0.2).tolist(),
            "b": np.arange(0.5, 1.0, 0.1).tolist()
        }
        # all_doc_ids = self.documents.keys()
        best_params = {"k1": 1.2, "b": 0.75}
        best_recall = 0.0
        
        total_experiments = len(list(ParameterGrid(param_grid)))
        experiment_count = 0
        
        for params in tqdm.tqdm(ParameterGrid(param_grid)):
            experiment_count += 1
            self.bm25 = BM25Okapi(self.tokenized_docs, k1=params["k1"], b=params["b"])
            correct_predictions = 0
            
            for data in train_data:
                question = data['tokenized_question']
                doc_ids = data['document_id']
                best_match_indices, best_matches = self.retrieve_datasets(question)
                if doc_ids in best_match_indices:
                    correct_predictions += 1
            recall = correct_predictions / len(train_data)
            
            if self.use_wandb:
                wandb.log({
                    "k1": params["k1"],
                    "b": params["b"],
                    "recall@5": recall,
                    "experiment_progress": experiment_count / total_experiments
                })
            
            if recall > best_recall:
                best_recall = recall
                best_params = params.copy()
                
                if self.use_wandb:
                    wandb.run.summary["best_k1"] = best_params["k1"]
                    wandb.run.summary["best_b"] = best_params["b"]
                    wandb.run.summary["best_recall@5"] = best_recall

            print(f"\nExperiment {experiment_count}/{total_experiments}: k1={params['k1']}, b={params['b']}, Recall@5={recall:.3f}")
        
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=best_params["k1"], b=best_params["b"])
        print(f"Best parameters: k1={best_params['k1']}, b={best_params['b']}, Recall@5={best_recall:.3f}")
        
        if self.use_wandb:
            wandb.config.update({
                "final_k1": best_params["k1"],
                "final_b": best_params["b"],
                "final_recall": best_recall
            })

    def generate_result_path(original_path):
        dir_name = os.path.dirname(original_path)
        base_name = os.path.basename(original_path)
        name, ext = os.path.splitext(base_name)
        new_base = f"{name}_result{ext}"
        return os.path.join(dir_name, new_base)

    def test_model(self, test_path="./data/test.jsonl"):
        test_data, len_test_data = get_data.load_datasets(test_path)
        result_path = self.generate_result_path(test_path)
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'w', encoding='utf-8') as f:
            for data in tqdm.tqdm(test_data):
                question = data['question']
                best_match_indices, best_matches = self.retrieve_datasets(question)
                f.write(json.dumps({
                    'question': question,
                    'document_id': best_match_indices
                }) + '\n')

'''
if __name__ == "__main__":
    kbqa = BM25_KBQA(file_path='./data/documents.jsonl', top_k=5, refine_model=True)

    if kbqa.use_wandb:
        wandb.finish()
'''