import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from langchain_core.documents import Document
from datetime import datetime, timedelta
from uuid import uuid4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import fitz
from skeleton_main import get_file_paths, process_pdf, flatten_docs, generate_uuids, vectorstore, embeddings, text_splitter

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

FILE_PATH = '/Users/hayan/airflow/dags/raw_pdf_1'
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
text_splitter = SemanticChunker(embeddings)


def process_pdf_files(ti, **kwargs):
    file_paths = get_file_paths(FILE_PATH)
    doc_600_100 = process_pdf(file_paths, embeddings, text_splitter)
    flattened_list = flatten_docs(doc_600_100)
    ti.xcom_push(key='flattened_list', value=flattened_list)

def add_documents_to_vectorstore(ti, **kwargs):
    flattened_list = ti.xcom_pull(key='flattened_list', task_ids='process_pdf_files')
    uuids = generate_uuids(len(flattened_list))
    vectorstore.add_documents(
        documents=flattened_list,
        embedding=embeddings,
        ids=uuids
    )


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 3, 20),
}

dag = DAG(
    'skeleton_rag',
    default_args=default_args,
    description='A DAG for processing PDFs, storing embeddings, and running QA pipeline.',
    catchup=False,
)

process_pdf_task = PythonOperator(
    task_id='process_pdf_files',
    python_callable=process_pdf_files,
    dag=dag,
)

add_to_vectorstore_task = PythonOperator(
    task_id='add_documents_to_vectorstore',
    python_callable=add_documents_to_vectorstore,
    retries=1,
    dag=dag,
)

# Task 순서 정의
process_pdf_task >> add_to_vectorstore_task