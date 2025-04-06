import BM25
import generator
import get_data
import json
import os
import tqdm

class KBQA:
    def __init__(self, retriever):
        if retriever=='BM25':
            self.retriever = BM25.BM25_KBQA(file_path='./data/documents.jsonl')
        self.generator_model = generator.AnswerGenerator()

    def generate_single_question_answer(self, question):
        tokenized_question = get_data.preprocess_text(question)
        top_indices, retrieved_docs = self.retriever.retrieve_single_question(tokenized_question)
        docs = self.get_docs(top_indices)
        answer = self.generator_model.generate_answer(question, docs)
        return answer
    
    def generate_datasets_answer(self, dataset_path='./data/val.jsonl', gold_file=True, use_wandb=False):
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb
            self.get_wandb_api_key()
        result_path = self.generate_result_path(dataset_path)
        if os.path.exists(result_path):
            os.remove(result_path)
        datas, len_data = get_data.load_datasets(dataset_path)
        with open(result_path, 'a', encoding='utf-8') as f:
            for data in tqdm.tqdm(datas, total=len_data, desc="Generating answers"):
                question = data['question']
                tokenized_question = get_data.preprocess_document(question)
                best_match_indices, best_matches = self.retriever.retrieve_dataset(tokenized_question)
                retrieved_docs = self.get_docs(best_match_indices)
                answer = self.generator_model.generate_answer(question, retrieved_docs)
                data['answer'] = answer
                data['document_id'] = best_match_indices
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        if gold_file:
            metrics = self.calculate_metrics(dataset_path, result_path)
            print(f"Metrics for {dataset_path}:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Recall@5: {metrics['recall@5']:.3f}")
            print(f"MRR@5: {metrics['mrr@5']:.3f}")

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

    def get_docs(self, top_indices, doc_path='./data/documents.jsonl'):
        with open(doc_path, 'r', encoding='utf-8') as f:
            documents = [json.loads(line) for line in f if line.strip()]
        retrieved_docs = [get_data.preprocess_text(documents[i]['document_text']) for i in top_indices]
        return retrieved_docs
    
    def calculate_metrics(self, gold_file, pred_file):
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_data = [json.loads(line) for line in f if line.strip()]
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = [json.loads(line) for line in f if line.strip()]
        
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
        
        metrics = {
            'accuracy': accuracy,
            'recall@5': recall_5,
            'mrr@5': mrr_5
        }
            
        return metrics
    
if __name__ == "__main__":
    kbqa = KBQA(retriever='BM25')

    # Example question
    # question = "when did the british first land in north america"
    # answer = kbqa.generate_single_question_answer(question)
    # print(f"Answer: {answer}")

    # Example validation
    kbqa.generate_datasets_answer(dataset_path='./data/val.jsonl', gold_file=True, use_wandb=False)