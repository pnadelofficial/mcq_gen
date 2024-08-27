import autogen
from autogen import AssistantAgent, UserProxyAgent
from langchain.text_splitter import TokenTextSplitter
import pandas as pd
import numpy as np
import pypdf
import os
import nltk
import prompts
import re
from openai import OpenAI
import random

nltk.download('punkt_tab')

class Dataloader:
    def __init__(self,
                 name) -> None:
        self.name = name
        self.base_path = f"./data/{name}"
        self.texts_with_metadata = []

    def _read_files_with_metadata(self, filename):
        if filename.endswith('.txt'):
            with open(filename, 'r') as f:
                text = f.read()
        else:
            pdf = pypdf.PdfReader(open(filename, 'rb'))
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        return {'title': filename, 'text': text}
    
    def __call__(self):
        for file in os.listdir(self.base_path):
            if file.startswith('cc_'):
                self.texts_with_metadata.append(self._read_files_with_metadata(os.path.join(self.base_path, file)))
        return self.texts_with_metadata

class Embedder:
    def __init__(self,
                 client,
                 embedding_model,
                 dataloader,
                 name,
                 chunk_size=200, 
                 chunk_overlap=50) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.dataloader = dataloader
        self.texts_with_metadata = self.dataloader()
        self.name = name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
    
    def _chunk_one(self, text, max_chunk_size=100):
        docs = self.text_splitter.split_text(text)
        sentences = nltk.sent_tokenize(text)
        chunks = []
        
        for doc in docs:
            chunk = []
            doc_sents = nltk.sent_tokenize(doc)
            
            if len(doc_sents) > 0:
                for s in sentences:
                    if doc_sents[0] in s:
                        chunk.append(s)
                        break
                chunk.extend(doc_sents[1:-1])  
                for s in sentences:
                    if doc_sents[-1] in s and s not in chunk:
                        chunk.append(s)
                        break
                full_chunk = ' '.join(chunk)
                if len(full_chunk.split()) > max_chunk_size:
                    trimmed_chunk = []
                    word_count = 0
                    for sent in chunk:
                        sent_words = sent.split()
                        if word_count + len(sent_words) <= max_chunk_size:
                            trimmed_chunk.append(sent)
                            word_count += len(sent_words)
                        else:
                            break
                    full_chunk = ' '.join(trimmed_chunk)
                chunks.append(full_chunk)
            else:
                chunks.append(doc)
        return chunks
    
    def _chunk(self):
        self.chunked_texts = []
        for doc in self.texts_with_metadata:
            chunks = self._chunk_one(doc['text'])
            for i, chunk in enumerate(chunks):
                self.chunked_texts.append({'title': doc['title'], 'text': chunk, 'chunk_id': i})
    
    def _embed(self, text):
        res = self.client.embeddings.create(input=text, model=self.embedding_model)
        return np.array(res.data[0].embedding)
    
    def _embed_docs(self):
        self.embedded_docs = []
        for doc in self.chunked_texts:
            embedding = self._embed(doc['text'])
            self.embedded_docs.append(embedding)
        self.embedded_docs = np.array(self.embedded_docs)
    
    def __call__(self):
        self._chunk()
        if os.path.exists(f"./data/{self.name}/embeddings.npy"):
            self.embedded_docs = np.load(f"./data/{self.name}/embeddings.npy")
            return self.embedded_docs
        else:        
            self._embed_docs()
            np.save(f"./data/{self.name}/embeddings.npy", self.embedded_docs)
            return self.embedded_docs
        # graph rag here

class TopicGenerator:
    def __init__(self,
                 name,
                 top_n,
                 topic_num,
                 embedder: Embedder,
                 client: OpenAI,
                 model_provider) -> None:
        self.name = name
        self.top_n = top_n
        self.topic_num = topic_num
        self.embedder = embedder
        self.client = client
        self.model_provider = model_provider
        self.model = 'gpt-4o-mini' if self.model_provider == 'OpenAI' else 'claude' ## TODO implement Claude (claudette: https://github.com/AnswerDotAI/claudette)
        self.embedded_docs = np.load(f"./data/{self.name}/embeddings.npy")

    def __call__(self):
        self.topics = []
        titles = [doc['title'].replace('cc_', '') for doc in self.embedder.texts_with_metadata]
        for title in titles:
            query_embedding = self.embedder._embed(title)
            chunk_ids = (self.embedded_docs @ query_embedding).argsort()[::-1][:self.top_n]
            transcript = '\n'.join([self.embedder.chunked_texts[i]['text'] for i in chunk_ids])
            print("len of topics", self.topic_num)
            prompt = prompts.TOPIC_GENERATION.format(title=title, transcript=transcript, topic_num=self.topic_num)
            messages = [
                {'role':'user', 'content':prompt}
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=random.uniform(0.8, 1.2)
            )
            raw = completion.choices[0].message.content
            topics = re.findall(r'\[.*?\]', raw, re.DOTALL)[0]
            topics = [t.replace('[', '').replace(']', '').strip() for t in topics.split('\n') if t.strip() != '']
            self.topics.extend(topics)
        return self.topics

class QuestionGenerator:
    def __init__(self,
                 name,
                 num_questions_each, 
                 model_provider,
                 retriver,
                 subject=None,
                 topics=None,
                 few_shot=False,
                 debug=None) -> None:
        self.name = name
        self.num_questions_each = num_questions_each
        self.retriever = retriver
        self.model_provider = model_provider
        self.subject = subject
        self.topics = topics
        self.few_shot = few_shot
        self.debug = debug

## extra column for citations

    def __call__(self, cb=None):
        DFs = []
        if self.debug:
            self.topics = self.topics[:self.debug] 
        for i, topic in enumerate(self.topics):
            print(f"Topic {i+1}/{len(self.topics)}")
            if cb:
                cb(i, topic, len(self.topics))
            self.mcqc = MCQChat(name=self.name, 
                                query=topic,
                                model_provider=self.model_provider,
                                num_questions=self.num_questions_each, 
                                retriver=self.retriever, 
                                few_shot=self.few_shot, 
                                subject=self.subject)
            qs = self.mcqc(topic.replace(',', '').strip())
            df = self.mcqc.to_df(qs)
            DFs.append(df)
        self.tot = pd.concat(DFs).reset_index(drop=True)
        return self.tot
    
    def to_csv(self, fn):
        self.tot.to_csv(fn)

class Retriever:
    def __init__(self,
                 client,
                 name,
                 embedder: Embedder,
                 embedding_model) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.name = name
        self.embedder = embedder
        self.embedded_docs = np.load(f"./data/{self.name}/embeddings.npy")
    
    def _embed(self, text):
        res = self.client.embeddings.create(input=text, model=self.embedding_model)
        return np.array(res.data[0].embedding)
    
    def __call__(self, query, k=5):
        query_embedding = self._embed(query)
        chunk_ids = (self.embedded_docs @ query_embedding).argsort()[::-1][:k]
        return [self.embedder.chunked_texts[i] for i in chunk_ids] ## TODO implement reranker: https://cookbook.openai.com/examples/search_reranking_with_cross-encoders

class MCQChat:
    def __init__(self,
                 name,
                 retriver,
                 query,
                 model_provider,
                 num_questions=3,
                 max_rounds=25,
                 few_shot=False,
                 subject=None) -> None:
        self.name = name
        self.retriever = retriver
        self.query = query
        self.model_provider = model_provider
        self.num_questions = num_questions
        self.max_rounds = max_rounds
        self.few_shot = few_shot
        self.subject = subject
        self.model = 'gpt-4o-mini' if self.model_provider == 'OpenAI' else 'claude' ## TODO implement Claude (claudette: https://github.com/AnswerDotAI/claudette)

        config_list = {'config_list': [{
                'model': self.model,
                'api_key': os.environ.get("OPENAI_API_KEY"),
                "temperature":random.uniform(0.8, 1.2)
            }]
        }

        if self.few_shot:
            ex_qs = []
            ex_question_docs = [f for f in os.listdir(f'./data/{self.name}/') if f.startswith('ex_')] 
            for f in ex_question_docs:
                with open(f'./data/{self.name}/{f}') as f:
                    ex_qs.append(f.read())
            ex_qs = '\n'.join(ex_qs)
            system_message = prompts.QUIZZER_PROMPT_AGENTS_FEW_SHOT.format(num_questions=self.num_questions, examples=ex_qs)
        else:
            system_message = prompts.QUIZZER_PROMPT_AGENTS.format(num_questions=self.num_questions)
        chunks = self.retriever(self.query, k=5)
        content_message = prompts.USER_PROMPT_AGENTS.format(question=query, summaries=chunks)

        system_message = system_message + '\n' + content_message

        self.question_generator = AssistantAgent("question_generator", llm_config=config_list, system_message=system_message)     
        self.question_revisor = AssistantAgent("question_revisor", llm_config=config_list, system_message=prompts.QUESTION_REVISOR_MESSAGE)
        self.question_editor = AssistantAgent("question_editor", llm_config=config_list, system_message=prompts.EDITOR_MESSAGE)
        self.accepter = AssistantAgent("accepter", llm_config=config_list, system_message=prompts.ACCEPTER_MESSAGE)
        self.user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding","use_docker": False,}, human_input_mode="NEVER")

        self.agent_list = [
            self.user_proxy,
            self.question_generator,
            self.question_revisor,
            self.question_editor,
            self.accepter
        ]

    def _init_graph_dict(self):
        self.graph_dict = {}
        self.graph_dict[self.user_proxy] = [self.question_generator]
        self.graph_dict[self.question_generator] = [self.question_revisor]
        self.graph_dict[self.question_revisor] = [self.question_editor]
        self.graph_dict[self.question_editor] = [self.accepter]
    
    def _init_groupchat(self):
        self.groupchat = autogen.GroupChat(
            agents=self.agent_list, 
            messages=[], 
            max_round=self.max_rounds, 
            allowed_or_disallowed_speaker_transitions=self.graph_dict, 
            allow_repeat_speaker=None, 
            speaker_transitions_type="allowed"
        )

        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )

    def _parse_question(self, question):
        stem = question.split('\n')[0]
        options = question.split('\n')[1:]
        if len(options) == 0:
            return
        if options[-1].startswith('D'):
            pass
        else:
            options = options[:4]
        parsed = [o for o in options if (o != '') and ((o.startswith('A')) or (o.startswith('Distractor_')))]
        while len(parsed) < 7:
            parsed.append('')
        return stem, *parsed
    
    def _start_chat(self, query):
        return self.user_proxy.initiate_chat(self.manager, message=query)

    def __call__(self, query):
        self._init_graph_dict()
        self._init_groupchat()
        self.o = self._start_chat(query)
        
        out_interim = self.o.chat_history[-1]['content'].split('\n\n')
        if out_interim[0].startswith('1'):
            out = '\n\n'.join(self.o.chat_history[-1]['content'].split('\n\n')[:-1])
        else:     
            out = '\n\n'.join(self.o.chat_history[-1]['content'].split('\n\n')[1:-1])
        out_list = re.split(r'(?=\d+\))', out)[1:]
        questions = [o.strip() for o in out_list]

        parsed = [self._parse_question(question) for question in questions]
        return parsed
    
    def to_df(self, qs):
        cols = [(re.sub(r'[A-z0-9]+\)', '',a) for a in list(q)) for q in qs if q]
        return pd.DataFrame(cols, columns=['question', 'altA', 'distractor1', 'distractor2', 'distactor3', 'distractor4', 'distractor5', 'distractor6'])

        