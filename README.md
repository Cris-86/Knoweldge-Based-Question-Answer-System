# Model 

The code is implemented in **Python** and UI is implemented in **streamlit**

The overall workflow is as follows:

<p align="center"> 
    <img src="img\workflow.png", width="50%">
</p> 

## Requirements
* beautifulsoup4
* colbert_ai
* gensim
* jsonlines
* nltk
* numpy
* rank_bm25
* Requests
* scikit_learn
* scipy
* tqdm
* wandb
* sentence_transformers
* bert_score
* streamlit

You can run 
```bash
pip install -r requirements.txt
```

## Usage
* To run this code, you should first create **credentials** and enter your api_key in the document. 
* If you want to use wandb to visualize the training process and record results, you should create **wandbKey**  and enter your wandb api_key in the document. 
* You can directly run streamlit in your CMD to get the UI interface and the corresponding KBQA system.

You can run 
```bash
streamlit run app.py
```

### UI interface
#### Selecting a search algorithm 

Choose whether to have a GPU based on the environment. **Note that this will affect the choice of hybrid algorithm**

w/o GPU hybrid algorithm is the combination of BM25 and word2vec. w/ GPU hybrid algorithm is the combination of BM25, word2vec and Colbert.

<p align="center"> 
    <img src=".\img\1.png">
</p>  

You can choose example question as input.

<p align="center"> 
    <img src=".\img\2.png">
</p> 

The final output wil include 3 parts, Generated answer, Summary of retrieved documents, Retrieved files:

<p align="center"> 
    <img src=".\img\3.png">
</p> 

<p align="center"> 
    <img src=".\img\4.png">
</p> 
