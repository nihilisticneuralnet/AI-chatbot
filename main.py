import torch
import numpy as np
from sentence_transformers import util
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import requests
import streamlit as st

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

hf_token =st.secrets['hf_token'] # insert hf api key

LLM_api_key=st.secrets['LLM_api_key'] # insert ai21 api key

st.title('AI Chatbot')

pdf = 'Corpus.pdf'

if pdf:
    pdf = PdfReader(pdf)
    text = ''
    for i in range(len(pdf.pages)):
        text += pdf.pages[i].extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    input = st.text_input(label='Ask your question here 😀')

    submit = st.button('Submit')

    if submit and input:
        try:
            st.session_state.conversation_history.append({'question': input, 'answer': ''})

            api_url = """https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2""" 
            headers = {"Authorization": f"Bearer {hf_token}"}

            def query(texts):
                response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
                return response.json()

            question = query([input])
            query_embeddings = torch.FloatTensor(question)

            output = query(chunks)
            output = torch.from_numpy(np.array(output)).to(torch.float)
            result = util.semantic_search(query_embeddings, output, top_k=2)

            final = [chunks[result[0][i]['corpus_id']] for i in range(len(result[0]))]

            context = ' '.join([entry['question'] + ' ' + entry['answer'] for entry in st.session_state.conversation_history]) + ' ' + ' '.join(final)

            url = "https://api.ai21.com/studio/v1/answer"
            payload = {
                "context": context,
                "question": input
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {LLM_api_key}"
            }

            response = requests.post(url, json=payload, headers=headers)
            answer = response.json().get('answer')

            if answer:
                st.session_state.conversation_history[-1]['answer'] = answer
                st.write(answer)
                st.header(':blue[Context]')
                st.write(final[0])
            else:
                st.error('The answer is not in the document ⚠️, kindly contact the business directly')
        except:
            st.error('The answer is not in the document ⚠️, kindly contact the business directly')

    st.header('Conversation History')
    for entry in st.session_state.conversation_history:
        st.write(f"**Q:** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")
