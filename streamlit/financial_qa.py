import json
import jsonlines
from operator import itemgetter
from datetime import datetime, timedelta
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import instructor

from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_upstage import UpstageEmbeddings
from langchain_milvus.vectorstores import Milvus
from langchain_community.vectorstores import Milvus as Mil2
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt
)
from langchain_teddynote.evaluator import GroundednessChecker
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, chain
from datetime import datetime, timedelta
from typing import List
import instructor
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rc


class TimeFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class SearchQuery(BaseModel):
    query: str
    time_filter: TimeFilter

class Label(BaseModel):
    chunk_id: int = Field(description="The unique identifier of the text chunk")
    chain_of_thought: str = Field(
        description="The reasoning process used to evaluate the relevance"
    )
    relevancy: int = Field(
        description="Relevancy score from 0 to 10, where 10 is most relevant",
        ge=0,
        le=10,
    )

class RerankedResults(BaseModel):
    labels: list[Label] = Field(description="List of labeled and ranked chunks")

    @field_validator("labels")
    @classmethod
    def model_validate(cls, v: list[Label]) -> list[Label]:
        return sorted(v, key=lambda x: x.relevancy, reverse=True)


class TimeProcessor:
    def __init__(self):
        self.today = datetime(2025, 1, 25)
    
    def adjust_time_filter_to_week(self, time_filter):
        """
        특정 날짜(YYYY-MM-DD)가 주어진 경우, 해당 날짜를 포함하는 주(월~일)의
        첫 번째 날(월요일)과 마지막 날(일요일)로 변환하는 함수.
        """
        start_date = time_filter.start_date
        end_date = time_filter.end_date

        if start_date is None or end_date is None:
            if start_date is not None and end_date is None:
                start_of_week = start_date - timedelta(days=start_date.weekday())
                end_of_week = start_of_week + timedelta(days=6)
                return {
                    "start_date": start_of_week.replace(hour=0, minute=0, second=0),
                    "end_date": end_of_week.replace(hour=23, minute=59, second=59)
                }
            elif end_date is not None and start_date is None:
                start_of_week = end_date - timedelta(days=end_date.weekday())
                end_of_week = start_of_week + timedelta(days=6)
                return {
                    "start_date": start_of_week.replace(hour=0, minute=0, second=0),
                    "end_date": end_of_week.replace(hour=23, minute=59, second=59)
                }
            else:
                return None

        if start_date.year == end_date.year and start_date.month==end_date.month and start_date.day==end_date.day:
            start_of_week = start_date - timedelta(days=start_date.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            return {
                "start_date": start_of_week.replace(hour=0, minute=0, second=0),
                "end_date": end_of_week.replace(hour=23, minute=59, second=59)
            }

        return {
            "start_date": start_date,
            "end_date": end_date
        }
    
    def get_query_date(self, question):
        days_since_last_friday = (self.today.weekday() - 4) % 7
        last_friday = self.today - timedelta(days=days_since_last_friday)
        issue_date = last_friday.strftime("%Y-%m-%d")

        client = instructor.from_openai(OpenAI())
        response = client.chat.completions.create(
            model="o1",
            response_model=SearchQuery,
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are an AI assistant that extracts date ranges from financial queries.
                    The current report date is {issue_date}.
                    Your task is to extract the relevant date or date range from the user's query
                    and format it in YYYY-MM-DD format.
                    If no date is specified, answer with None value.
                    """,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )

        parsed_dates = self.adjust_time_filter_to_week(response.time_filter)

        if parsed_dates:
            start = parsed_dates['start_date']
            end = parsed_dates['end_date']
        else:
            start = None
            end = None

        if start is None or end is None:
            expr = None
        else:
            expr = f"issue_date >= '{start.strftime('%Y%m%d')}' AND issue_date <= '{end.strftime('%Y%m%d')}'"
        
        return expr


class DocumentProcessor:
    def save_docs_to_jsonl(self, documents, file_path):
        with jsonlines.open(file_path, mode="w") as writer:
            for doc in documents:
                writer.write(doc.dict())
    
    def load_documents(self, filepath):
        splitted_doc = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('\n('):
                    continue
                data = json.loads(line)
                doc = Document(
                    page_content=data['page_content'],
                    metadata=data['metadata']
                )
                splitted_doc.append(doc)
        return splitted_doc
    
    def convert_to_list(self, example):
        if isinstance(example["contexts"], list):
            contexts = example["contexts"]
        else:
            try:
                contexts = json.loads(example["contexts"])
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {example['contexts']} - {e}")
                contexts = []
        return {"contexts": contexts}
    
    def format_docs(self, docs):
        return "\n\n".join(
            f"Issue Date: {doc.metadata.get('issue_date', 'Unknown')}\nContent: {doc.page_content}"
            for doc in docs
        )


class Reranker:
    def rerank_results(self, query, chunks):
        client = instructor.from_openai(OpenAI())
        return client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=RerankedResults,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert search result ranker. Your task is to evaluate the relevance of each text chunk to the given query and assign a relevancy score.

                    For each chunk:
                    1. Analyze its content in relation to the query.
                    2. Provide a chain of thought explaining your reasoning.
                    3. Assign a relevancy score from 0 to 10, where 10 is most relevant.

                    Be objective and consistent in your evaluations.
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    {{ query }}
                    
                    {% for chunk in chunks %}
                    {{ chunk.text }}
                    {% endfor %}
                    
                    Please provide a RerankedResults object with a Label for each chunk.
                    """,
                },
            ],
            context={"query": query, "chunks": chunks},
        )
    
    def reranking(self, docs, question, k=15):
        chunks = [{
            "id": idx, 
            "issue_date": doc.metadata['issue_date'], 
            "text": doc.page_content
        } for idx, doc in enumerate(docs)]
        
        documents_with_metadata = [{
            "text": doc.page_content, 
            "metadata": doc.metadata
        } for doc in docs]
        
        reranked_results = self.rerank_results(query=question, chunks=chunks)

        chunk_dict = {chunk["id"]: chunk["text"] for chunk in chunks}
        top_k_results = [
            chunk_dict.get(label.chunk_id, "") 
            for label in reranked_results.labels[:k] 
            if label.chunk_id in chunk_dict
        ]

        reranked_results_with_metadata = []
        for reranked_result in top_k_results:
            page_content = reranked_result
            matching_metadata = None
            
            for doc in documents_with_metadata:
                if doc["text"] == page_content:
                    matching_metadata = doc["metadata"]
                    break

            document = Document(
                metadata=matching_metadata,
                page_content=page_content
            )
            reranked_results_with_metadata.append(document)

        return reranked_results_with_metadata


class EmbeddingManager:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(
            model='solar-embedding-1-large-query',
        )


class VectorStoreManager:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        self.uri = 'http://127.0.0.1:19530'
        self.setup_vectorstores()
    
    def setup_vectorstores(self):
        self.vectorstore_text = Milvus(
            embedding_function=self.embedding_manager.embeddings,
            connection_args={'uri': self.uri},
            index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
            collection_name='text_semantic_per_80_00_test'
        )
        
        self.vectorstore_predict = Milvus(
            embedding_function=self.embedding_manager.embeddings,
            connection_args={'uri': self.uri},
            index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
            collection_name='text_semantic_per_80_00'
        )
        
        self.vectorstore_table = Milvus(
            embedding_function=self.embedding_manager.embeddings,
            connection_args={'uri': self.uri},
            index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
            collection_name='table_v7'
        )
        
        self.vectorstore_raptor = Mil2(
            embedding_function=self.embedding_manager.embeddings,
            connection_args={'uri': self.uri},
            index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
            collection_name='raptor_v3'
        )
        
        self.vectorstore_image = Milvus(
            embedding_function=self.embedding_manager.embeddings,
            connection_args={'uri': self.uri},
            index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
            collection_name='image_v4'
        )
    
    def get_retriever(self, store_type, search_kwargs=None):
        stores = {
            'text': self.vectorstore_text,
            'predict': self.vectorstore_predict,
            'table': self.vectorstore_table,
            'raptor': self.vectorstore_raptor,
            'image': self.vectorstore_image
        }
        
        if store_type not in stores:
            raise ValueError(f"Unknown store type: {store_type}")
            
        if search_kwargs is None:
            search_kwargs = {'k': 20 if store_type != 'image' else 3}
            
        return stores[store_type].as_retriever(search_kwargs=search_kwargs)


class PromptManager:
    def __init__(self):
        self.text_prompt = PromptTemplate.from_template(
            '''
            Today is '2025-01-25'. 
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Answer in Korean. Answer in detail.

            #Question:
            {question}
            #Context:
            {context}

            #Answer:'''
        )
        
        self.predict_prompt = PromptTemplate.from_template(
            '''You are future-predicting expert AI chatbot about financial.
            주어진 정보는 retrieved context들이야. 이 정보를 바탕으로 미래를 예측해줘.

            If one of the table or text says it doesn't know or it can't answer, don't mention with that.
            주어진 예측을 한 근거도 함께 자세히 설명해줘. 왜 그런 예측을 어떤 걸 근거로 내놓았는지 알려줘.
            Don't answer with the specific numbers.

            #Question:
            {question}

            #Context:
            {context}
            '''
        )
        
        self.table_prompt = PromptTemplate.from_template(
            '''You are an assistant for question-answering tasks.
            Use the following pieces of retrieved table to answer the question.
            If you don't know the answer, just say that you don't know.
            Answer in Korean. Answer in detail.

            #Question:
            {question}
            #Context:
            {context}

            #Answer:'''
        )
        
        self.general_prompt = PromptTemplate.from_template(
            '''You are question-answering AI chatbot about financial reports.
            주어진 정보는 retrieved context들이야. 이 정보를 바탕으로 질문에 대해 자세히 설명해줘.

            If one of the table or text says it doesn't know or it can't answer, don't mention with that.
            And some questions may not be answered simply with context, but rather require inference. In those cases, answer by inference.

            #Question:
            {question}

            #Context:
            {context}
            '''
        )
        
        self.prompt_raptor = PromptTemplate.from_template(
            '''You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Answer in Korean. Answer in detail.
            If the context mentions an unrelated date, do not mention that part.
            Summarize and organize your answers based on the various issues that apply to the period.

            #Question:
            {question}
            #Context:
            {context}

            #Answer:'''
        )
        
        self.prompt_routing = PromptTemplate.from_template(
            '''주어진 사용자 질문을 `날짜`, `호수`, `일반` 중 하나로 분류하세요. 한 단어 이상으로 응답하지 마세요.
            If user question has the expression about date, route it to the '날짜' datasource.

            {question}

            Classification:'''
        )
        
        self.prompt_routing_2 = PromptTemplate.from_template(
            '''You are an expert at routing a user question to the appropriate data source.

            If the user is asking for a brief summary, route it to the '요약' datasource.
            If the user is asking for more detailed or general information, route it to the '일반' datasource.
            If the user is asking for some prediction, route it to the '예측' datasource.

            Just answer with one word of datasource.

            Today is January 25th, 2025. Only classify as predictions asking about things after today.

            {question}

            datasource:'''
        )

        self.prompt_routing_3 = PromptTemplate.from_template(
            '''You are an expert at routing a user question to the appropriate data source.

            If the user is asking for some prediction, route it to the '예측' datasource.
            If the user is asking for more detailed or general information, route it to the '일반' datasource.

            Just answer with one word of datasource.

            Today is January 25th, 2025. Only classify as predictions asking about things after today.

            {question}

            datasource:'''
        )


class ChainManager:
    def __init__(self, vector_store_manager, time_processor, reranker, document_processor, prompt_manager):
        self.vector_store_manager = vector_store_manager
        self.time_processor = time_processor
        self.reranker = reranker
        self.document_processor = document_processor
        self.prompt_manager = prompt_manager
        
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        self.llm_text = ChatOpenAI(model='o1', temperature=1)
        self.llm_general = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        
        self.question_answer_relevant = GroundednessChecker(
            llm=ChatOpenAI(model='gpt-4o-mini', temperature=0), 
            target='question-answer'
        ).create()
        
        self.query_retrieval_relevant = GroundednessChecker(
            llm=ChatOpenAI(model='gpt-4o-mini', temperature=0), 
            target='question-retrieval'
        ).create()
        
        # Load documents and set up BM25 retrievers
        self.setup_bm25_retrievers()
        
        # Set up chains
        self.text_chain = self.create_text_chain()
        self.predict_chain = self.create_predict_chain()
        self.text_chain_2 = self.create_text_chain_2()
        self.table_chain = self.create_table_chain()
        self.date_chain = self.create_date_chain()
        self.general_chain = self.create_general_chain()
        self.raptor_chain = self.create_raptor_chain()
        self.raptor_date_chain = self.create_raptor_date_chain()

    def setup_bm25_retrievers(self):
        text_docs = self.document_processor.load_documents('./chunked_jsonl/250313_text_semantic_per_80.jsonl')
        table_docs = self.document_processor.load_documents('./chunked_jsonl/table_v7.jsonl')
        
        self.bm25_retriever_text = KiwiBM25Retriever.from_documents(text_docs)
        self.bm25_retriever_text.k = 20
        
        self.bm25_retriever_table = KiwiBM25Retriever.from_documents(table_docs)
        self.bm25_retriever_table.k = 5
    
    def kill_table(self, result):
        if self.question_answer_relevant.invoke({'question': result['question'], 'answer': result['text']}).score == 'no':
            print('no_kill')
            result['context'] = self.table_chain.invoke({'question': result['question']})
        else:
            result['context'] = result['text']
        return result
    
    def create_text_chain(self):
        return (
            RunnableParallel(
                question=itemgetter('question')
            ).assign(expr = lambda x: self.time_processor.get_query_date(x['question'])
            ).assign(context_raw=lambda x: RunnableLambda(
                lambda _: self.vector_store_manager.get_retriever(
                    'text', search_kwargs={'expr': x['expr'], 'k': 25}
                ).invoke(x['question'])
            ).invoke({}),
            ).assign(
                context=lambda x: self.reranker.reranking(
                    list({doc.metadata.get("pk"): doc for doc in (x['context_raw'])}.values()),
                    x['question'], 15
                )
            ).assign(
                formatted_context=lambda x: self.document_processor.format_docs(x['context'])
            )
            | RunnableLambda(
                lambda x: {
                    "question": x['question'],
                    "context": x['formatted_context'], 
                }
            )
            | self.prompt_manager.text_prompt
            | self.llm_text
            | StrOutputParser()
        )
    
    def create_predict_chain(self):
        predict_expression = 'issue_date >= "20241224" AND issue_date <="20250124"'
        return (
            RunnableParallel(
                question=itemgetter('question')
            ).assign(context=lambda x: RunnableLambda(
                lambda _: self.vector_store_manager.get_retriever(
                    'predict', search_kwargs={'k': 20, 'expr':predict_expression}
                ).invoke(x['question'])
            ).invoke({}),
            ).assign(
                formatted_context=lambda x: self.document_processor.format_docs(x['context'])
            )
            | RunnableLambda(
                lambda x: {
                    "question": x['question'],
                    "context": x['formatted_context'], 
                }
            )
            | self.prompt_manager.predict_prompt
            | self.llm_text
        )
    
    def create_text_chain_2(self):
        return (
            RunnableParallel(
                question=itemgetter('question')
            ).assign(expr = lambda x: self.time_processor.get_query_date(x['question'])
            ).assign(milvus=lambda x: RunnableLambda(
                lambda _: self.vector_store_manager.get_retriever(
                    'text', search_kwargs={'k': 25}
                ).invoke(x['question'])
            ).invoke({}),
            bm25=lambda x: self.bm25_retriever_text.invoke(x['question'])
            ).assign(
                context=lambda x: self.reranker.reranking(
                    list({doc.metadata.get("pk"): doc for doc in (x['milvus'] + x['bm25'])}.values()),
                    x['question'], 20
                )
            ).assign(
                formatted_context=lambda x: self.document_processor.format_docs(x['context'])
            )
            | RunnableLambda(
                lambda x: {
                    "question": x['question'],
                    "context": x['formatted_context'], 
                }
            )
            | self.prompt_manager.text_prompt
            | self.llm_text
            | StrOutputParser()
        )
    
    def create_table_chain(self):
        return (
            RunnableParallel(
                question=itemgetter('question')
            ).assign(expr = lambda x: self.time_processor.get_query_date(x['question'])
            ).assign(milvus=lambda x: RunnableLambda(
                lambda _: self.vector_store_manager.get_retriever(
                    'table', search_kwargs={'expr': x['expr'], 'k': 10}
                ).invoke(x['question'])
            ).invoke({}),
            bm25_raw=lambda x: self.bm25_retriever_table.invoke(x['question'])
            ).assign(
                bm25_filtered=lambda x: [
                    doc for doc in x["bm25_raw"]
                    if not x["expr"] or (
                        x["expr"].split("'")[1] <= doc.metadata.get("issue_date", "") <= x["expr"].split("'")[3]
                    )
                ],
            ).assign(
                context=lambda x: list({
                    doc.metadata.get("pk"): doc 
                    for doc in (x['milvus'] + x['bm25_filtered'])
                }.values())
            ).assign(
                formatted_context=lambda x: self.document_processor.format_docs(x['context'])
            )
            | RunnableLambda(
                lambda x: {
                    "question": x['question'],
                    "context": x['formatted_context'],
                }
            )
            | self.prompt_manager.text_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def create_date_chain(self):
        return (
            RunnableParallel(
                question=itemgetter('question'),
                text=self.text_chain,
            )
            | RunnableLambda(lambda x: self.kill_table(x))
            | self.prompt_manager.general_prompt
            | self.llm_general
        )
    
    def create_general_chain(self):
        return (
            RunnableParallel(
                question=itemgetter('question'),
                text=self.text_chain_2,
            )
            | RunnableLambda(lambda x: self.kill_table(x))
            | self.prompt_manager.general_prompt
            | self.llm_general
        )
    
    def create_raptor_chain(self):
        metadata_field_info = [
            AttributeInfo(
                name='source',
                description='문서의 번호. 네 자리의 숫자와 "호"로 이루어져 있다. 현재 1090호부터 1120호까지 존재한다.',
                type='string',
            ),
        ]

        prompt_query = get_query_constructor_prompt(
            'summary of weekly financial report about bonds',
            metadata_field_info
        )

        output_parser = StructuredQueryOutputParser.from_components()

        query_llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)
        query_constructor = prompt_query | query_llm | output_parser
        
        retriever_raptor = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=self.vector_store_manager.vectorstore_raptor,
            structured_query_translator=MilvusTranslator(),
            search_kwargs={'k': 10}
        )
        
        return (
            RunnableParallel(
                question=itemgetter('question')
            ).assign(expr = lambda x: self.time_processor.get_query_date(x['question'])
            ).assign(context=lambda x: retriever_raptor.invoke(x['question']))
            | RunnableLambda(
                lambda x: {
                    "question": x['question'],
                    "context": x['context'],
                }
            )
            | self.prompt_manager.prompt_raptor
            | self.llm
        )
    
    def create_raptor_date_chain(self):
        return (
            RunnableParallel(
                question=itemgetter('question')
            ).assign(expr = lambda x: self.time_processor.get_query_date(x['question'])
            ).assign(context=lambda x: RunnableLambda(
                lambda _: self.vector_store_manager.get_retriever(
                    'raptor', search_kwargs={'expr': x['expr'], 'k': 10}
                ).invoke(x['question'])
            ).invoke({})
            )
            | RunnableLambda(
                lambda x: {
                    "question": x['question'],
                    "context": x['context'],
                }
            )
            | self.prompt_manager.prompt_raptor
            | self.llm
        )


class Router:
    def __init__(self, chain_manager, prompt_manager):
        self.chain_manager = chain_manager
        self.prompt_manager = prompt_manager
        
        self.chain_routing = (
            {'question': RunnablePassthrough()}
            | self.prompt_manager.prompt_routing
            | ChatOpenAI(model='o1')
            | StrOutputParser()
        )
        
        self.chain_routing_2 = (
            {'question': RunnablePassthrough()}
            | self.prompt_manager.prompt_routing_2
            | ChatOpenAI(model='gpt-4o-mini')
            | StrOutputParser()
        )
        
        self.chain_routing_3 = (
            {'question': RunnablePassthrough()}
            | self.prompt_manager.prompt_routing_3
            | ChatOpenAI(model='gpt-4o-mini')
            | StrOutputParser()
        )

        self.full_chain = (
            {'topic': self.chain_routing, 'question': itemgetter('question')}
            | RunnableLambda(self.route)
            | StrOutputParser()
        )
    
    def route_3(self, info):
        if '예측' in info['topic'].lower():
            print('predict_chain')
            return self.chain_manager.predict_chain
        else:
            print('general_chain')
            return self.chain_manager.general_chain
    
    def route_2(self, info):
        if '요약' in info['topic'].lower():
            print('raptor_date_chain')
            return self.chain_manager.raptor_date_chain
        elif '예측' in info['topic'].lower():
            print('predict_chain')
            return self.chain_manager.predict_chain
        else:
            print('date_chain')
            return self.chain_manager.date_chain
    
    def route(self, info):
        if '날짜' in info['topic'].lower():
            info['topic'] = self.chain_routing_2.invoke(info['question'])
            return self.route_2(info)
        elif '호수' in info['topic'].lower():
            print('raptor_chain')
            return self.chain_manager.raptor_chain
        else:
            info['topic'] = self.chain_routing_3.invoke(info['question'])
            return self.route_3(info)


class FinancialQASystem:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager(self.embedding_manager)
        self.time_processor = TimeProcessor()
        self.document_processor = DocumentProcessor()
        self.reranker = Reranker()
        self.prompt_manager = PromptManager()
        self.chain_manager = ChainManager(
            self.vector_store_manager,
            self.time_processor,
            self.reranker,
            self.document_processor,
            self.prompt_manager
        )
        self.router = Router(self.chain_manager, self.prompt_manager)
    
    def ask(self, question):
        expr = self.time_processor.get_query_date(question)
        answer = self.router.full_chain.invoke({'question': question})
        print(answer)
        
        rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False
        context = self.vector_store_manager.get_retriever('image', {'k': 3}).invoke(question, expr=expr)
        
        for i in context:
            rar = self.chain_manager.query_retrieval_relevant.invoke({'context': i, 'question': question})
            if rar.score == 'yes':
                plt.title('참고 자료')
                image_path = i.metadata['image'].replace('raw_pdf_copy3', 'parsed_pdf')
                img = Image.open(image_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
        return answer
