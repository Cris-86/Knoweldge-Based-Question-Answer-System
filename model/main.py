import generator
import get_data
import json
import os

import tqdm
import numpy as np
from bert_score import score

current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)
data_path = os.path.join(script_dir, "data", "documents.jsonl")

class KBQA:
    def __init__(self, file_path=data_path, 
                 retriever='BM25', 
                 top_k=5, 
                 sentence_search=False,
                 use_GPU=False):
        
        self.use_GPU = use_GPU
        self.top_k = top_k
        self.sentence_search = sentence_search
        self.file_path = file_path
        '''
        if use_GPU:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPU is not available. Please set use_GPU=False.")
        '''
        if retriever=='BM25':
            import BM25
            self.retrieverName = 'BM25'
            self.retriever = BM25.BM25_KBQA(file_path=self.file_path, top_k=self.top_k)
        elif retriever=='Word2Vec':
            import word2vec
            self.retrieverName = 'Word2Vec'
            self.retriever = word2vec.Word2Vec_KBQA(file_path=self.file_path, top_k=self.top_k)
        elif retriever=='Hybrid' and use_GPU==False:
            import BM25
            import word2vec
            self.retrieverName = 'Hybrid'
            self.bm25_retriever = BM25.BM25_KBQA(file_path=self.file_path, top_k=self.top_k)
            self.w2v_retriever = word2vec.Word2Vec_KBQA(file_path=self.file_path, top_k=self.top_k)
        elif retriever=='Hybrid' and use_GPU==True:
            import BM25
            import word2vec
            import ColBERT_KBQA
            self.retrieverName = 'Hybrid'
            self.bm25_retriever = BM25.BM25_KBQA(file_path=self.file_path, top_k=self.top_k)
            self.w2v_retriever = word2vec.Word2Vec_KBQA(file_path=self.file_path, top_k=self.top_k)
            self.ColBERT_retriever = ColBERT_KBQA.ColBERT_KBQA(
                file_path=self.file_path, 
                top_k=self.top_k,
                enable_sentence_search=self.sentence_search
            )
        elif retriever=='ColBERT':
            import ColBERT_KBQA
            self.retrieverName = 'ColBERT'
            self.retriever = ColBERT_KBQA.ColBERT_KBQA(
                file_path=self.file_path, 
                top_k=self.top_k,
                enable_sentence_search=self.sentence_search
            )
        self.generator_model = generator.AnswerGenerator()
    
    def score_regularization(self, scores):
        scores = np.array(scores)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return scores
    
    def combine_ranking(self, bm25_indices, bm25_scores, w2v_indices, w2v_scores):
        bm25_dict = dict(zip(bm25_indices, bm25_scores))
        w2v_dict = dict(zip(w2v_indices, w2v_scores))

        all_indices = set(bm25_dict.keys()).union(set(w2v_dict.keys()))

        combined_scores = {}
        for idx in all_indices:
            score_bm25 = bm25_dict.get(idx, 0)
            score_w2v = w2v_dict.get(idx, 0)
            combined_score = (score_bm25 + score_w2v) / 2
            combined_scores[idx] = combined_score

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        sorted_indices = [item[0] for item in sorted_items]
        sorted_scores = [item[1] for item in sorted_items]
        
        return sorted_indices, sorted_scores

    def generate_single_question_answer(self, question):
        tokenized_question = get_data.preprocess_document(question)
        
        if self.retrieverName == 'Hybrid' and self.use_GPU==False:
            bm25_indices, bm25_scores = self.bm25_retriever.retrieve_single_question(tokenized_question)
            w2v_indices, w2v_scores = self.w2v_retriever.retrieve_single_question(tokenized_question)
            bm25_scores = self.score_regularization(bm25_scores)
            w2v_scores = self.score_regularization(w2v_scores)
            sorted_indices, sorted_scores = self.combine_ranking(bm25_indices, bm25_scores, w2v_indices, w2v_scores)
            top_indices = sorted_indices
            top_indices = list(dict.fromkeys(top_indices))
            docs = self.get_docs(top_indices)

        elif self.retrieverName == 'Hybrid' and self.use_GPU==True:
            bm25_indices, bm25_scores = self.bm25_retriever.retrieve_single_question(tokenized_question)
            w2v_indices, w2v_scores = self.w2v_retriever.retrieve_single_question(tokenized_question)
            colbert_indices, colbert_scores = self.ColBERT_retriever.retrieve_single_question(question)
            bm25_scores = self.score_regularization(bm25_scores)
            w2v_scores = self.score_regularization(w2v_scores)
            colbert_scores = self.score_regularization(colbert_scores)
            sorted_indices, sorted_scores = self.combine_ranking(bm25_indices, bm25_scores, w2v_indices, w2v_scores)
            final_indices, final_scores = self.combine_ranking(sorted_indices, sorted_scores, colbert_indices, colbert_scores)
            top_indices = sorted_indices
            top_indices = list(dict.fromkeys(top_indices))
            docs = self.get_docs(top_indices)

        elif self.retrieverName == 'ColBERT':
            if self.sentence_search:
                top_indices, relevant_sentences = self.retriever.retrieve_with_sentence_search(
                    question, enable_fine_search=True
                )
                docs = self.get_docs(top_indices)
                if relevant_sentences:
                    docs = ['. '.join(relevant_sentences)]
            else:
                top_indices, _ = self.retriever.retrieve_single_question(question)
                docs = self.get_docs(top_indices)
        else:
            top_indices, _ = self.retriever.retrieve_single_question(tokenized_question)
            docs = self.get_docs(top_indices)
            
        answer, summary = self.generator_model.generate_answer(question, docs, return_summary=True)
        return answer, top_indices, docs, summary
    
    def refine_sentence_top_k(self):
        sentence_top_ks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
                           55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                           105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
                           155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
        import wandb
        self.get_wandb_api_key()
        wandb.login(key=self.wandb_api_key)
        wandb.init(project="NLP_assignment", 
                    config={
                            "model": self.retrieverName,
                            "dataset_path": './data/val.jsonl',
                            "task": "KBQA",
                            "description": "Refine sentence top k for KBQA System"
                    }) 
        total_experiments = len(sentence_top_ks)
        experiment_count = 0
        for sentence_top_k in sentence_top_ks:
            experiment_count += 1
            metrics = self.generate_datasets_answer(dataset_path='./data/val.jsonl', 
                                                   gold_file=True, 
                                                   use_wandb=False, 
                                                   batch_size=10, 
                                                   sentence_top_k=sentence_top_k, 
                                                   refine=True)
            
            wandb.log({
                        "sentence_top_k_value": sentence_top_k,
                        "accuracy": metrics['accuracy'],
                        "bert_score": metrics['bert_score'],
                        "recall@5": metrics['recall@5'],
                        "mrr@5": metrics['mrr@5'],
                        "experiment_progress": experiment_count / total_experiments
                    })


    def generate_datasets_answer(self, 
                                 dataset_path='./data/val.jsonl', 
                                 gold_file=True, 
                                 use_wandb=False, 
                                 batch_size=10, 
                                 sentence_top_k=19, 
                                 refine=False, 
                                 temperature_1=0.1, 
                                 temperature_2=0.8):
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb
            self.get_wandb_api_key()
            wandb.login(key=self.wandb_api_key) 
            wandb.init(project="NLP_assignment", 
                    config={
                            "model": self.retrieverName,
                            "dataset_path": dataset_path,
                            "task": "KBQA",
                            "description": f"{self.retrieverName} model for KBQA System"
                    }) 
        
        result_path = self.generate_result_path(dataset_path)
        if os.path.exists(result_path):
            os.remove(result_path)
            
        datas, len_data = get_data.load_datasets(dataset_path)
        with open(result_path, 'a', encoding='utf-8') as f:
            for i in tqdm.tqdm(range(0, len_data, batch_size), desc="Processing batches"):
                end_idx = min(i + batch_size, len_data)
                batch_data = datas[i:end_idx]
                
                for data in tqdm.tqdm(batch_data, total=end_idx-i, desc="Generating answers", leave=False):
                    question = data['question']
                    tokenized_question = get_data.preprocess_document(question)
                    data['tokenized_question'] = tokenized_question
                    if self.retrieverName == 'Hybrid' and self.use_GPU==False:
                        bm25_indices, bm25_scores = self.bm25_retriever.retrieve_datasets(tokenized_question)
                        w2v_indices, w2v_scores = self.w2v_retriever.retrieve_datasets(tokenized_question)
                        bm25_scores = self.score_regularization(bm25_scores)
                        w2v_scores = self.score_regularization(w2v_scores)
                        sorted_indices, sorted_scores = self.combine_ranking(bm25_indices, bm25_scores, w2v_indices, w2v_scores)
                        top_indices = sorted_indices
                        best_match_indices = list(dict.fromkeys(top_indices))
                       
                    elif self.retrieverName == 'Hybrid' and self.use_GPU==True:
                        bm25_indices, bm25_scores = self.bm25_retriever.retrieve_datasets(tokenized_question)
                        w2v_indices, w2v_scores = self.w2v_retriever.retrieve_datasets(tokenized_question)
                        colbert_indices, colbert_scores = self.ColBERT_retriever.retrieve_datasets(question)
                        bm25_scores = self.score_regularization(bm25_scores)
                        w2v_scores = self.score_regularization(w2v_scores)
                        colbert_scores = self.score_regularization(colbert_scores)
                        sorted_indices, sorted_scores = self.combine_ranking(bm25_indices, bm25_scores, w2v_indices, w2v_scores)
                        final_indices, final_scores = self.combine_ranking(sorted_indices, sorted_scores, colbert_indices, colbert_scores)
                        top_indices = final_indices
                        best_match_indices = list(dict.fromkeys(top_indices))
                    
                    elif self.retrieverName == 'ColBERT':
                        if self.sentence_search:
                            best_match_indices, relevant_sentences = self.retriever.retrieve_with_sentence_search(
                                question, enable_fine_search=True, sentence_top_k=sentence_top_k
                            )
                        else:
                            best_match_indices, _ = self.retriever.retrieve_datasets(question)
                    else:
                        best_match_indices, _ = self.retriever.retrieve_datasets(data['tokenized_question'])
                    
                    retrieved_docs = self.get_docs(best_match_indices)
                    if self.sentence_search:
                        if relevant_sentences:
                            retrieved_docs = ['. '.join(relevant_sentences)]
                    
                    answer = self.generator_model.generate_answer(question, retrieved_docs, temperature_1, temperature_2)
                    
                    output = {
                        'question': question,
                        'answer': answer,
                        'document_id': best_match_indices
                    }
                    f.write(json.dumps(output) + '\n')    

        if gold_file:
            metrics = self.calculate_metrics(dataset_path, result_path)
            print(f"Metrics for {dataset_path}:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"BERT Score: {metrics['bert_score']:.3f}")
            print(f"Recall@5: {metrics['recall@5']:.3f}")
            print(f"MRR@5: {metrics['mrr@5']:.3f}")
            if self.use_wandb:
                wandb.log({
                    "accuracy": metrics['accuracy'],
                    "bert_score": metrics['bert_score'],
                    "recall@5": metrics['recall@5'],
                    "mrr@5": metrics['mrr@5']
                })
            if refine:
                return metrics  

    def refine_temperature(self):
        temperature = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        import wandb
        self.get_wandb_api_key()
        wandb.login(key=self.wandb_api_key)
        wandb.init(project="NLP_assignment",
                    config={
                            "model": self.retrieverName,
                            "dataset_path": './data/val.jsonl',
                            "task": "KBQA",
                            "description": "Refine temperature for KBQA System"
                    })
        total_experiments = len(temperature) * len(temperature)
        experiment_count = 0
        for temp_1 in temperature:
            for temp_2 in temperature:
                experiment_count += 1
                metrics = self.generate_datasets_answer(dataset_path='./data/val.jsonl', 
                                                       gold_file=True, 
                                                       use_wandb=False, 
                                                       batch_size=10, 
                                                       refine=True,
                                                       temperature_1=temp_1, 
                                                       temperature_2=temp_2)
                
                wandb.log({
                            "temperature_1": temp_1,
                            "temperature_2": temp_2,
                            "accuracy": metrics['accuracy'],
                            "bert_score": metrics['bert_score'],
                            "recall@5": metrics['recall@5'],
                            "mrr@5": metrics['mrr@5'],
                            "experiment_progress": experiment_count / total_experiments
                        })

    def generate_result_path(self, original_path):
        dir_name = os.path.dirname(original_path)
        base_name = os.path.basename(original_path)
        name, ext = os.path.splitext(base_name)
        new_base = f"{name}_result{ext}"
        return os.path.join(dir_name, new_base)

    def get_wandb_api_key(self, file_path='wandbKey'):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.wandb_api_key = f.read().strip()
        else:
            raise FileNotFoundError(f"Credentials file '{file_path}' not found.")

    def get_docs(self, top_indices, doc_path=data_path):
        with open(doc_path, 'r', encoding='utf-8') as f:
            documents = [json.loads(line) for line in f if line.strip()]
        retrieved_docs = [get_data.preprocess_text(documents[i]['document_text']) for i in top_indices]
        return retrieved_docs
    
    def calculate_metrics(self, gold_file, pred_file):
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_data = [json.loads(line) for line in f if line.strip()]
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = [json.loads(line) for line in f if line.strip()]

        gold_answers = [gold['answer'].lower() for gold in gold_data]
        pred_answers = [pred['answer'].lower() for pred in pred_data]
        P, R, F1 = score(pred_answers, gold_answers, lang="en", verbose=False)
        bert_scores = F1.tolist()

        total = len(gold_data)
        correct = 0
        recall_5_sum = 0
        mrr_5_sum = 0
        
        for gold, pred in zip(gold_data, pred_data):
            if gold['answer'].lower() == pred['answer'].lower():
                correct += 1

            true_doc_id = gold['document_id']
            pred_doc_ids = pred['document_id']
            
            # Recall@5
            if true_doc_id in pred_doc_ids:
                recall_5_sum += 1
                
            # MRR@5
            if true_doc_id in pred_doc_ids:
                rank = pred_doc_ids.index(true_doc_id) + 1
                mrr_5_sum += 1.0 / rank
        
        accuracy = correct / total
        recall_5 = recall_5_sum / total
        mrr_5 = mrr_5_sum / total
        bert_score_avg = sum(bert_scores) / total
        
        metrics = {
            'accuracy': accuracy,
            'bert_score': bert_score_avg,
            'recall@5': recall_5,
            'mrr@5': mrr_5
        }
            
        return metrics


if __name__ == "__main__":
    # source /etc/network_turbo

    # Example question
    kbqa = KBQA(retriever='Word2Vec', use_GPU=True)
    question = "when did the british first land in north america"
    answer, top_indices, docs, summary = kbqa.generate_single_question_answer(question)
    print(f"Top indices: {top_indices}")
    docs = [doc.replace('\n', ' ') for doc in docs]
    print(f"Documents: {docs}")
    print(f"Summary: {summary}")
    print(f"Answer: {answer}")
    # question = "when did the 1st world war officially end"
    # answer, top_indices = kbqa.generate_single_question_answer(question)
    # print(f"Answer: {answer}")
    # print(f"Top indices: {top_indices}")

    # Example validation
    # Test  dataset Samples
    # kbqa = KBQA(retriever='ColBERT', sentence_search=True)
    # kbqa.refine_sentence_top_k()

    # kbqa = KBQA(retriever='Hybrid', use_GPU=True)
    # kbqa.generate_datasets_answer(dataset_path='./data/val.jsonl', gold_file=True, use_wandb=True)
    # kbqa.refine_temperature()
    # kbqa.generate_datasets_answer(dataset_path='./data/val.jsonl', gold_file=True, use_wandb=False)

'''

if __name__ == "__main__":
    # kbqa = KBQA(retriever='ColBERT', sentence_search=True)
    kbqa = KBQA(retriever='ColBERT')
    question = "when did the british first land in north america"
    answer, top_indices = kbqa.generate_single_question_answer(question)
    print(f"Answer: {answer}")
    print(f"Top indices: {top_indices}")
'''