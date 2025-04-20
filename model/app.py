from flask import Flask, request, jsonify
from flask_cors import CORS
import main

# 创建Flask应用实例
app = Flask(__name__)

# 启用CORS支持
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# 定义路由
@app.route('/ask', methods=['POST'])

def ask_question():
    data = request.json
    question = data.get('question', '')
    algorithm = data.get('algorithm', '')
    answer = generate_answer(question, algorithm)
    documents = retrieve_documents(question, algorithm)
    return jsonify({'answer': answer, 'documents': documents})

def generate_answer(question, algorithm):
    if algorithm == 'BM25':
        kbqa = main.KBQA(retriever='BM25')
        answer, top_indices, docs, summary = kbqa.generate_single_question_answer(question)
    elif algorithm == 'Word2Vec':
        kbqa = main.KBQA(retriever='Word2Vec')
        answer, top_indices, docs, summary = kbqa.generate_single_question_answer(question)
    elif algorithm == 'Hybrid-NoGPU':
        kbqa = main.KBQA(retriever='Hybrid', use_GPU=False)
        answer, top_indices, docs, summary = kbqa.generate_single_question_answer(question)
    elif algorithm == 'Hybrid-GPU':
        kbqa = main.KBQA(retriever='Hybrid', use_GPU=True)
        answer, top_indices, docs, summary = kbqa.generate_single_question_answer(question)
    elif algorithm == 'ColBERT':
        kbqa = main.KBQA(retriever='ColBERT')
        answer, top_indices, docs, summary = kbqa.generate_single_question_answer(question)
    return f"这是基于{algorithm}算法生成的答案。"

def retrieve_documents(question, algorithm):
    # 模拟检索文档的逻辑
    return ['Document 1', 'Document 2', 'Document 3']

if __name__ == '__main__':
    # 启动Flask应用
    app.run(host='0.0.0.0', port=8080, debug = True)