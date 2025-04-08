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
            self.bm25 = BM25Okapi(self.tokenized_docs, k1=2.4, b=0.9)
    
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

    def retrieve_single_question(self, question):
        tokenized_question = get_data.preprocess_document(question)
        scores = self.bm25.get_scores(tokenized_question)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:self.top_k]
        best_matches = [self.documents[i] for i in top_indices]
        return top_indices,  best_matches

    def retrieve_datasets(self, tokenized_question):
        scores = self.bm25.get_scores(tokenized_question)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:self.top_k]
        best_matches = [self.documents[i] for i in top_indices]
        return top_indices,  best_matches
    
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
                best_match_indices, best_matches = self.retrieve_datasets_bm25(question)
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