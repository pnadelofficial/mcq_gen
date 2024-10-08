import streamlit as st
import os
from mcq_gen import QuestionGenerator
import utils
from datetime import datetime
from openai import OpenAI
import collections
from functools import partial
import subprocess
import prompts

st.title("Multiple Choice Question Generation")

if st.session_state.get('CLIENT') is None:
    st.session_state['CLIENT'] = OpenAI(api_key=st.secrets['openai']["open_ai_key"])
    
name = st.text_input("Enter your course title", key='name_input')
if (st.session_state.get('name') is None) or (st.session_state.get('name') != name):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(BASE_DIR, "data", name)
    os.makedirs(data_dir, exist_ok=True)
    st.session_state['name'] = name

if st.session_state.get('current_question') is None:
    st.session_state['completed_questions'] = []

if (st.session_state.get('convo_messages') is None) or (st.session_state.get('name') != name):
    convo_sys_prompt = prompts.CONVO_SYS_PROMPT.format(course_title=name)   
    print(convo_sys_prompt)
    st.session_state['convo_messages'] = [
        {"role": "system", "content": convo_sys_prompt},
        {"role": "assistant", "content": "Hello! I'm here to help you generate multiple choice questions. How can I help you?"}
    ]

convo = st.checkbox("Use conversation model", key='convo_check')
if st.session_state.get("expander_state") is None:
    st.session_state["expander_state"] = True
if convo:
    utils.conversation()
else:
    if st.session_state.get('summary') is None:
        st.session_state['summary'] = False

num_questions_help = """
In addition to the total number of questions, you can specify the number of questions per topic. 
These topics are generated from the course content you upload and are used to generate questions.
The value you enter here will be used to divide the total number of questions you want to generate over all of the generated topics.
""".strip()

num_questions_total = st.number_input("Enter total number of questions", min_value=1, max_value=100)
num_questions = st.number_input("Enter number of questions per topic", min_value=1, max_value=10, help=num_questions_help)
topic_num = num_questions_total // num_questions

cc_uploaded_files = st.file_uploader(
    "Upload course content", accept_multiple_files=True
)

fw_check = st.checkbox("Use example questions", key='fw_check')
if fw_check:
    st.warning("""
        Before you upload example questions, please make sure that each question is formatted as follows:
            1) Each question is followed by a new line.
            2) The answer choices are followed by a new line.
            3) The correct answer is wrapped by two asterisks.
            4) Questions should be separated by two new lines.
    """.strip())
    ex_uploaded_files = st.file_uploader(
        "Upload example questions", accept_multiple_files=True
    )

for uploaded_file in cc_uploaded_files:
    with open(f"./data/{name}/cc_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

if fw_check:
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
            topics = utils.get_topics(name, topic_num=topic_num)
            st.session_state['topics'] = topics

    if st.session_state.get('gen_questions_expanded') is None:
        st.session_state['gen_questions_expanded'] = True
    with st.status('Generating questions...', expanded=st.session_state['gen_questions_expanded']):
        if st.session_state.get('ret') is None:
            ret = utils.get_retriever(name)
            st.session_state['ret'] = ret        
        pbar = st.progress(0, text='Writing questions...')
        few_shot = True if fw_check > 0 else False
        print(st.session_state['summary'])
        qg = QuestionGenerator(name=name, 
                            num_questions_each=num_questions,
                            retriver=st.session_state['ret'],
                            model_provider="OpenAI",
                            topics=st.session_state['topics'],
                            few_shot=few_shot,
                            summary=st.session_state['summary'],
                            debug=topic_num)
        questions = qg(cb=partial(utils.pbar_callback, pbar=pbar))
        pbar.progress(1.0, text='Questions written!')
        st.session_state['ran_questions'] = True
    utils.toggle_expander()

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
    if (st.session_state.get('csv') is None) or (csv != st.session_state.get('csv')):
        st.session_state['csv'] = csv
    if st.session_state.get('csv') is not None:
        st.download_button(
            label="Download CSV",
            data=st.session_state['csv'],
            file_name=f"{name}_mcqs_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            mime="text/csv"
        )

if st.button("Start Over"):
    st.session_state.clear()
    subprocess.run(["rm", "-rf", f"./data/{name}"])
    st.rerun()

st.markdown("<footer><small>Assembed by Peter Nadel | Tufts University | Tufts Technology Services | Reserch Technology</small></footer>", unsafe_allow_html=True) 