# AI Chatbot
## Requirements
```
Python 3.7+
Streamlit
PyPDF2
LangChain
Sentence-Transformers
Torch
Requests
```
```
pip install streamlit PyPDF2 langchain sentence-transformers[train]==3.0.1 torch requests
```
## How to Run the Code
#### 1. Clone the Repository
```
git clone https://github.com/nihilisticneuralnet/AI-chatbot.git
cd <repository_directory>
```
#### 2. Insert your tokens

replace ```st.secrets['hf_token'] ``` with your hugging face api key: *SignUp in [huggingface](https://huggingface.co/) and make sure the permission for the api key is* **WRITE**   \
replace ```st.secrets['LLM_api_key'] ``` with your AI21 api key: *SignUp in [AI21](https://www.ai21.com/) to create the api key*


#### 3. Run the Streamlit App
```
streamlit run main.py
```
or 
```
python -m streamlit run main.py
```
![image](https://github.com/user-attachments/assets/67dd760d-dd75-42df-a159-d842f36ba2f3)
