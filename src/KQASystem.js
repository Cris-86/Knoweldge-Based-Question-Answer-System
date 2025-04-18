import React, { useState } from 'react';

function KQASystem() {
  const [question, setQuestion] = useState('');
  const [algorithm, setAlgorithm] = useState('Word2Vec');
  const [retrievedDocuments, setRetrievedDocuments] = useState([]);
  const [answer, setAnswer] = useState('');

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
  };

  const handleAlgorithmChange = (e) => {
    setAlgorithm(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // 模拟从后台获取数据
    const documents = ['Document 1', 'Document 2', 'Document 3', 'Document 4', 'Document 5'];
    const generatedAnswer = `This is the generated answer based on the ${algorithm} algorithm.`;

    // 更新状态
    setRetrievedDocuments(documents);
    setAnswer(generatedAnswer);
  };

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', padding: '20px' }}>
      <h1>Knowledge Question Answering (KQA) System</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <input
            type="text"
            placeholder="Enter your question"
            value={question}
            onChange={handleQuestionChange}
            style={{ width: '80%', padding: '10px', marginRight: '10px' }}
          />
          <button type="submit" style={{ padding: '10px 20px' }}>Submit</button>
        </div>
        <div style={{ marginTop: '20px' }}>
          <label>Choose Search Algorithm:</label>
          <select value={algorithm} onChange={handleAlgorithmChange} style={{ marginLeft: '10px', padding: '5px' }}>
            <option value="Word2Vec">Word2Vec</option>
            <option value="BM25">BM25</option>
            <option value="ColBert">ColBert</option>
            <option value="Hybrid">Hybrid</option>
          </select>
        </div>
      </form>
      {retrievedDocuments.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h3>Retrieved Documents:</h3>
          <ul>
            {retrievedDocuments.map((doc, index) => (
              <li key={index}>{doc}</li>
            ))}
          </ul>
        </div>
      )}
      {answer && (
        <div style={{ marginTop: '20px' }}>
          <h3>Answer:</h3>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default KQASystem;