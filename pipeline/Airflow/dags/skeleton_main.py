import os
import re
import itertools
import fitz
from langchain.document_loaders import PyMuPDFLoader
from langchain.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus.vectorstores import Milvus
from uuid import uuid4

from langchain.schema.runnable import RunnableParallel

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

FILE_PATH = '/Users/hayan/airflow/dags/raw_pdf_1/'
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
text_splitter = SemanticChunker(embeddings)

def is_numeric_or_negative(line):
    valid_chars = "0123456789-,. /%"
    return all(char in valid_chars for char in line.strip())

def pdf_preprocessing(doc_test):

    index = 0
    doc_to_kill = [0, 1]
    doc_to_save = []
    doc_to_die = []

    for j in doc_test:
        a = j.page_content
        b = []

        if 'Appendix' in j.page_content:
            doc_to_die.append(index)
            break
        for line in a.split('\n'):
            if re.search(r'^(표|그림|그래프|차트)\s*\d+', line.strip()):
                continue
            if line.startswith('자료 :') or line.startswith('(단위'):
                continue
            if is_numeric_or_negative(line)==True:
                continue
            if len(line) > 25 and not re.match(r'^(그림|표|주\))', line.strip()):
                b.append(line)

        c = '\n'.join(b).strip()
        j.page_content = c
        index += 1

    index = 0
    for j in doc_test:
        if len(j.page_content) < 50:
            doc_to_kill.append(index)
        index += 1

    for i in range(len(doc_test)):
        if i in doc_to_die:
            break
        if i not in doc_to_kill:
            doc_to_save.append(doc_test[i])

    return doc_to_save


def extract_issue_date(metadata):
    """
    'source' 필드에서 (YYYYMMDD) 형식의 날짜를 추출하여 YYYYMMDD 그대로 반환
    """
    source = metadata.get("source", "")
    match = re.search(r'\((\d{8})\)', source)  # 파일명에서 (YYYYMMDD) 패턴 찾기
    if match:
        return match.group(1)  # "20241220" 같은 날짜 그대로 반환 (date로 감지되지 않음)
    return None

def get_file_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

def generate_uuids(num_uuids):
    """주어진 개수만큼 UUID를 생성하는 함수."""
    return [str(uuid4()) for _ in range(num_uuids)]

def process_pdf(file_paths, embeddings, text_splitter):
    doc_600_100 = []
    
    for path in file_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        processed_docs = pdf_preprocessing(docs)
        split_doc = text_splitter.split_documents(processed_docs)

        for doc in split_doc:
            extracted_date = extract_issue_date(doc.metadata)
            if extracted_date:
                doc.metadata["issue_date"] = extracted_date

        doc_600_100.append(split_doc)
    
    return doc_600_100

def flatten_docs(doc_600_100):
    """문서 목록을 평탄화하는 함수."""
    return list(itertools.chain(*doc_600_100))

# 벡터스토어
URI = 'http://127.0.0.1:19530/'

vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={'uri': URI},
    index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
    collection_name='airflow_test1'
)


file_paths = get_file_paths(FILE_PATH)
doc_600_100 = process_pdf(file_paths, embeddings, text_splitter)
flattened_list = flatten_docs(doc_600_100)

uuids = generate_uuids(len(flattened_list))

vectorstore.add_documents(
    documents=flattened_list,
    embedding=embeddings,
    ids=uuids
)

def create_llm(temperature=0, model="gpt-4o-mini"):
    return ChatOpenAI(temperature=temperature, model=model)

def create_bm25_retriever(documents, k=10):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    return bm25_retriever

def create_prompt_template():
    return PromptTemplate.from_template(
        '''You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        When you answer, read the context and the context before that, and answer it with comparison and synthesis.
        Answer in Korean.

        #Question:
        {question}
        #Context:
        {context}

        #Answer:'''
    )

def create_chain(bm25_retriever, prompt, llm):
    return (
        RunnableParallel(
            context=bm25_retriever,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )