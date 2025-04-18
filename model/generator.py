import requests
import os
import BM25
import get_data
import json

class AnswerGenerator:
    def __init__(self, file_path='credentials', fine_generation=False):
        self.fine_generation = fine_generation
        self.get_api_key(file_path)
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }        
        with open("prompt.txt", "r") as f:
            self.prompt_template = f.read()
    
    def get_api_key(self, file_path='credentials'):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.api_key = f.read().strip()
        else:
            raise FileNotFoundError(f"Credentials file '{file_path}' not found.")

    def get_incontext_examples(self, question, train_path='./data/train.jsonl'):
        if self.fine_generation:
            # need refine
            kbqa = BM25.BM25_KBQA(file_path=train_path, top_k=2, refine_model=False, use_wandb=False)
            top_indices,  best_matches = kbqa.retrieve_single_question(question)
            retrieved_docs = [best_matches[i] for i in top_indices]
        else:
            retrieved_docs = []
            with open(train_path, 'r') as f:
                for i in range(2):
                    line = f.readline()
                    data = json.loads(line)
                    retrieved_docs.append(data)
        prompt = self.prompt_template.replace("{example_question_1}", retrieved_docs[0]["question"])
        prompt = prompt.replace("{example_answer_1}", retrieved_docs[0]["answer"])
        prompt = prompt.replace("{example_question_2}", retrieved_docs[1]["question"])
        prompt = prompt.replace("{example_answer_2}", retrieved_docs[1]["answer"])
        # print(f"Prompt: {prompt}")
        return prompt

    def generate_answer(self, question, current_context):
        len_context = len(current_context)
        context = "".join(current_context[:len_context])
        prompt = self.get_incontext_examples(question)
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{context}", context)
        payload = {
            "model": "Qwen/Qwen2-7B-Instruct",
            "stream": False,
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "messages": [
                {
                    "content": prompt,
                    "role": "user"
                }
            ]
        }
        try:
            response = requests.request("POST", url=self.url, json=payload, headers=self.headers)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Request failed: {str(e)}")
            return "[Error] API request failed"

'''
if __name__ == "__main__":
    retrieved_docs = [
        "Pom Klementieff is a French actress best known for playing Mantis in the Guardians of the Galaxy films...",
        "The character Mantis in the Marvel Cinematic Universe was portrayed by Pom Klementieff since her first appearance in Guardians of the Galaxy Vol. 2...",
        "Guardians of the Galaxy casting details: Chris Pratt as Star-Lord, Zoe Saldana as Gamora, and Pom Klementieff as Mantis..."
    ]
    
    question = "Who plays Mantis in Guardians of the Galaxy?"
    generator = AnswerGenerator()
    
    answer = generator.generate_answer(question, retrieved_docs)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
'''