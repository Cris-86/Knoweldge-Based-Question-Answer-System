import BM25
import generator
import get_data
import json
import os
import tqdm
import word2vec
import ColBERT_KBQA

class KBQA:
    def __init__(self, retriever='BM25', fine_search=False, top_k=5, sentence_search=False):
        self.top_k = top_k
        self.fine_search = fine_search
        self.sentence_search = sentence_search
        
        if retriever=='BM25':
            self.retrieverName = 'BM25'
            self.retriever = BM25.BM25_KBQA(file_path='./data/documents.jsonl')
        elif retriever=='Word2Vec':
            self.retrieverName = 'Word2Vec'
            self.retriever = word2vec.Word2Vec_KBQA()
        elif retriever=='Hybrid':
            self.retrieverName = 'Hybrid'
            self.bm25_retriever = BM25.BM25_KBQA()
            self.w2v_retriever = word2vec.Word2Vec_KBQA()
        elif retriever=='ColBERT':
            self.retrieverName = 'ColBERT'
            self.retriever = ColBERT_KBQA.ColBERT_KBQA(
                file_path='./data/documents.jsonl', 
                top_k=top_k,
                enable_sentence_search=sentence_search
            )
        self.generator_model = generator.AnswerGenerator()

    def generate_single_question_answer(self, question):
        tokenized_question = get_data.preprocess_document(question)
        
        if self.retrieverName == 'Hybrid':
            bm25_indices, _ = self.bm25_retriever.retrieve_single_question(tokenized_question)
            w2v_indices, _ = self.w2v_retriever.retrieve_single_question(tokenized_question)
            top_indices = list(dict.fromkeys(bm25_indices + w2v_indices))
            docs = self.get_docs(top_indices)
        elif self.retrieverName == 'ColBERT':
            if self.sentence_search and self.fine_search:
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
            
        answer = self.generator_model.generate_answer(question, docs)
        print(f"docs: {docs}")
        return answer, top_indices
    
    def generate_datasets_answer(self, dataset_path='./data/val.jsonl', gold_file=True, use_wandb=False, batch_size=10):
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
                            "description": f"{self.retrieverName} model for KBQA System, fine_search={self.fine_search}"
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
                    if self.retrieverName == 'Hybrid':
                        bm25_indices, _ = self.bm25_retriever.retrieve_datasets(data['tokenized_question'])
                        w2v_indices, _ = self.w2v_retriever.retrieve_datasets(data['tokenized_question'])
                        best_match_indices = list(dict.fromkeys(bm25_indices + w2v_indices))[:self.top_k]
                    elif self.retrieverName == 'ColBERT':
                        if self.sentence_search and self.fine_search:
                            best_match_indices, relevant_sentences = self.retriever.retrieve_with_sentence_search(
                                question, enable_fine_search=True
                            )
                        else:
                            best_match_indices, _ = self.retriever.retrieve_datasets(question)
                    else:
                        best_match_indices, _ = self.retriever.retrieve_datasets(data['tokenized_question'])
                    
                    retrieved_docs = self.get_docs(best_match_indices)
                    if self.sentence_search and self.fine_search:
                        if relevant_sentences:
                            retrieved_docs = ['. '.join(relevant_sentences)]
                    
                    answer = self.generator_model.generate_answer(question, retrieved_docs)
                    
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
            print(f"Recall@5: {metrics['recall@5']:.3f}")
            print(f"MRR@5: {metrics['mrr@5']:.3f}")
            if self.use_wandb:
                wandb.log({
                    "accuracy": metrics['accuracy'],
                    "recall@5": metrics['recall@5'],
                    "mrr@5": metrics['mrr@5']
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
    # source /etc/network_turbo

    # kbqa = KBQA(retriever='ColBERT', fine_search=True, sentence_search=True)

    # Example question
    # question = "when did the 1st world war officially end"
    # answer, top_indices = kbqa.generate_single_question_answer(question)
    # print(f"Answer: {answer}")
    # print(f"Top indices: {top_indices}")

    # Example validation
    # kbqa.generate_datasets_answer(dataset_path='./data/val.jsonl', gold_file=True, use_wandb=True)
    # kbqa.generate_datasets_answer(dataset_path='./data/val.jsonl', gold_file=True, use_wandb=True)

    kbqa = KBQA(retriever='ColBERT')
    kbqa.generate_datasets_answer(dataset_path='./data/val.jsonl', gold_file=True, use_wandb=True)

'''
if __name__ == "__main__":
    # kbqa = KBQA(retriever='ColBERT', fine_search=True, sentence_search=True)
    kbqa = KBQA(retriever='ColBERT')
    question = "when did the british first land in north america"
    answer, top_indices = kbqa.generate_single_question_answer(question)
    print(f"Answer: {answer}")
    print(f"Top indices: {top_indices}")
'''