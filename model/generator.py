import requests
import os

class AnswerGenerator:
    def __init__(self, file_path='credentials'):
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

    def generate_answer(self, question, current_context):
        len_context = len(current_context)
        context = "".join(current_context[:len_context])
        prompt = self.prompt_template.replace("{question}", question)
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