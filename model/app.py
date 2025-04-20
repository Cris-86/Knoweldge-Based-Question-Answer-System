from flask import Flask, request, jsonify
from flask_cors import CORS

# 创建Flask应用实例
app = Flask(__name__)

# 启用CORS支持
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

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
    # 生成答案的逻辑
    return f"这是基于{algorithm}算法生成的答案。"

def retrieve_documents(question, algorithm):
    # 模拟检索文档的逻辑
    return ['Document 1', 'Document 2', 'Document 3']

if __name__ == '__main__':
    # 启动Flask应用
    app.run(host='0.0.0.0', port=8080, debug = True)