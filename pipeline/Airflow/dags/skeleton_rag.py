import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from langchain_core.documents import Document
from datetime import datetime, timedelta
from uuid import uuid4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import fitz
from skeleton_main import get_file_paths, preprocessing, open_jsonl, generate_uuids, create_vectorstore, add_docs

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

FILE_PATH = '/Users/hayan/airflow/dags/raw_pdf'
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
text_splitter = SemanticChunker(embeddings)


def process_pdf_files(ti, **kwargs):
    preprocessing(FILE_PATH)
    text = open_jsonl(get_file_paths(FILE_PATH)[0].replace('.pdf', '_semantic_80.jsonl'))
    table = open_jsonl(get_file_paths(FILE_PATH)[0].replace('.pdf', '_table_v7.jsonl'))
    image = open_jsonl(get_file_paths(FILE_PATH)[0].replace('.pdf', '_image.jsonl'))
    raptor = open_jsonl(get_file_paths(FILE_PATH)[0].replace('.pdf', '_raptor.jsonl'))

    ti.xcom_push(key='text', value=text)
    ti.xcom_push(key='table', value=table)
    ti.xcom_push(key='image', value=image)
    ti.xcom_push(key='raptor', value=raptor)

def add_documents_to_vectorstore(ti, **kwargs):
    text = ti.xcom_pull(key='text', task_ids='process_pdf_files')
    table = ti.xcom_pull(key='table', task_ids='process_pdf_files')
    image = ti.xcom_pull(key='image', task_ids='process_pdf_files')
    raptor = ti.xcom_pull(key='raptor', task_ids='process_pdf_files')
    
    vectorstore_text = create_vectorstore('airflow-text')
    vectorstore_table = create_vectorstore('airflow-table')
    vectorstore_image = create_vectorstore('airflow-image')
    vectorstore_raptor = create_vectorstore('airflow-raptor')

    add_docs(vectorstore_text, text)
    add_docs(vectorstore_table, table)
    add_docs(vectorstore_image, image)
    add_docs(vectorstore_raptor, raptor)


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 3, 25),
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
