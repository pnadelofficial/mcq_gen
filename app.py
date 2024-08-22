import streamlit as st
import os
from mcq_gen import QuestionGenerator
import utils
from datetime import datetime
from openai import OpenAI
import collections

st.title("Multiple Choice Question Generation")

## chat for question preferences

if st.session_state.get('CLIENT') is None:
    st.session_state['CLIENT'] = OpenAI()

name = st.text_input("Enter your course title")
if st.session_state.get('name') is not None:
    os.makedirs(f"./data/{name}", exist_ok=True)
    st.session_state['name'] = name

num_questions_total = st.number_input("Enter total number of questions", min_value=1, max_value=100)
num_questions = st.number_input("Enter number of questions per topic", min_value=1, max_value=10)

cc_uploaded_files = st.file_uploader(
    "Upload course content", accept_multiple_files=True
)
ex_uploaded_files = st.file_uploader(
    "Upload example questions", accept_multiple_files=True
)

for uploaded_file in cc_uploaded_files:
    with open(f"./data/{name}/cc_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

for uploaded_file in ex_uploaded_files:
    with open(f"./data/{name}/ex_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

st.session_state['ran_questions'] = False
if st.button("Generate MCQs"):
    dl = utils.get_dl(name)

    with st.status('Reading uploaded documents...'):
        if st.session_state.get('emb') is None:
            emb = utils.save_emb(name, dl)
            st.session_state['emb'] = emb

    with st.status('Generating topics...'):
        if st.session_state.get('topics') is None:
            topics = utils.get_topics(name)
            st.session_state['topics'] = topics

    with st.status('Generating questions...'):
        if st.session_state.get('ret') is None:
            ret = utils.get_retriever(name)
            st.session_state['ret'] = ret        
        qg = QuestionGenerator(name=name, 
                            num_questions_each=num_questions,
                            retriver=st.session_state['ret'],
                            model_provider="OpenAI",
                            topics=st.session_state['topics'],
                            few_shot=True,
                            debug=num_questions_total if num_questions_total < len(st.session_state['topics']) else None) # TODO need to deal with when num_questions_total > len(topics)*num_questions :o
        questions = qg()
        st.session_state['ran_questions'] = True

if (st.session_state.get('questions') is None) and (st.session_state['ran_questions']):
    st.session_state['questions'] = questions
    
    if len(st.session_state['questions']) > 0:
        st.success("Questions generated successfully!")
    else:
        st.error("Failed to generate questions. Please try again.")    

if st.session_state.get('new_questions') is None:
    st.session_state['new_questions'] = collections.defaultdict(dict)
if st.session_state.get('new_answers') is None:
    st.session_state['new_answers'] = collections.defaultdict(dict)
if st.session_state.get('new_distractors') is None:    
    st.session_state['new_distractors'] = collections.defaultdict(dict)

if st.session_state.get('questions') is not None:
    qe = utils.QuestionEditor(st.session_state['questions'])
    csv = qe()
    if st.session_state.get('csv') is None:
        st.session_state['csv'] = csv
    if st.session_state.get('csv') is not None:
        st.download_button(
            label="Download CSV",
            data=st.session_state['csv'],
            file_name=f"{name}_mcqs_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            mime="text/csv"
        )