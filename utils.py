import streamlit as st  
import pandas as pd
from mcq_gen import Dataloader, Embedder, TopicGenerator, Retriever
import os 
# import app

os.environ["OPENAI_API_KEY"] = st.secrets['openai']["open_ai_key"]

@st.cache_data
def get_dl(name):
    dl = Dataloader(name)
    dl()
    return dl

def save_emb(name, dl):
    emb = Embedder(name=name, 
                   client=st.session_state['CLIENT'], 
                   embedding_model="text-embedding-3-small",
                   dataloader=dl)
    emb()
    return emb

def get_topics(name, topic_num):
    tg = TopicGenerator(name=name,
                        client=st.session_state['CLIENT'], 
                        top_n=5,
                        topic_num=topic_num,
                        embedder=st.session_state['emb'],
                        model_provider="OpenAI")
    topics = tg()
    return topics

def get_retriever(name):
    retriever = Retriever(name=name, 
                          client=st.session_state['CLIENT'],
                          embedder=st.session_state['emb'],
                          embedding_model="text-embedding-3-small")
    return retriever

class QuestionEditor:
    def __init__(self, df):
        self.df = df    
        self.edit_questions()
    
    @st.fragment
    def edit_question(self, i, row):
        with st.expander(f"### Question {i+1}", expanded=False):
            st.session_state['new_distractors'][i] = []
            
            skip_cb = st.checkbox("Skip question", key=f'skip_{i}')
            if skip_cb:
                st.session_state['new_questions'][i] = False
            
            question = st.text_area("Question", row["question"], key=f'question_ta_{i}')
            use_question = st.toggle("Use this question", False, key=f'question_{i}')
            if use_question:
                st.session_state['new_questions'][i] = question

            answer = st.text_area("Answer", row["altA"], key=f'answer_ta_{i}')
            use_answer = st.toggle("Use this answer", False, key=f'answer_{i}')
            if use_answer:
                st.session_state['new_answers'][i] = answer

            for j, distractor in enumerate(self.df.columns[2:]):
                distractor = st.text_area(distractor, row[distractor], key=f'ta_{i}_{j}')
                use = st.toggle("Use this distractor", False, key=f'toggle_{i}_{j}')
                if use:
                    st.session_state['new_distractors'][i].append(distractor)
            
    def edit_questions(self):
        with st.form("edit_questions_form"):
            for index, row in self.df.iterrows():
                self.edit_question(index, row)
                st.divider()
            submit_btn = st.form_submit_button("Submit")
            if submit_btn:
                self.make_new_df()

    def make_new_df(self):
        new_df = pd.DataFrame(columns=['Question', 'altA', 'altB', 'altC', 'altD'])
        for i in range(len(self.df)):
            if st.session_state['new_questions'][i]:
                question = st.session_state['new_questions'][i]
                answer = st.session_state['new_answers'][i]
                distractors = st.session_state['new_distractors'][i]
                new_row = [question, answer] + distractors
                try:
                    new_df.loc[i] = new_row
                    st.session_state['new_df'] = new_df.reset_index(drop=True)
                except ValueError:
                    st.error(f"Error with question {i+1}. Please make sure all questions have a question, an answer and three distractors.")
    
    def __call__(self):
        if st.session_state.get('new_df') is not None:
            st.write(st.session_state['new_df'])
            @st.cache_data
            def convert_df(df):
                return df.to_csv().encode("utf-8")
            csv = convert_df(st.session_state['new_df'])
            return csv

def pbar_callback(i, topic, topic_len, pbar):
    if topic.strip():
        pbar.progress((i/topic_len), text=f"Writing questions on {topic.replace(',','')}")
    else:
        pbar.progress(0, text="Writing questions...")

def toggle_expander():
    st.session_state['gen_questions_expanded'] = not st.session_state['gen_questions_expanded']