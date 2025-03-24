# 🕵️ Bogosa
### [AIFFELthon DS 3기] 보고사: 금융 보고서 기반 VectorDB 구축 및 RAG 최적화
- 금융 보고서 기반 최적 VectorDB 생성 & RAG 기법 최적화 및 신뢰도 높은 금융 특화 LLM 환경 구축 Project

📍 시연 영상 첨부 📍

<br>
<br>

## 팀원 및 역할
<table>
  <tr>
    <td>
        <a href="https://github.com/highlevelnotes">
         <img src="https://avatars.githubusercontent.com/u/124477843?v=4" width="150px" />  
        </a>
    </td>
    <td>
        <a href="https://github.com/hayannn">
         <img src="https://avatars.githubusercontent.com/u/102213509?v=4" width="150px" />  
        </a>
    </td>
    <td>
        <a href="https://github.com/gyumin-k">
         <img src="https://avatars.githubusercontent.com/u/61161614?v=4" width="150px" />  
        </a>
    </td>
  </tr>
  <tr>
    <td><p align="center"><b>이지우</b></p></td>
    <td><p align="center"><b>이하얀</b></p></td>
    <td><p align="center"><b>김규민</b></p></td>
  </tr>
  <tr>
    <td><p align="center"><a href="https://github.com/highlevelnotes">highlevelnotes</a></p></td>
    <td><p align="center"><a href="https://github.com/hayannn">hayannn</a></p></td>
    <td><p align="center"><a href="https://github.com/gyumin-k">gyumin-k</a></p></td>
  </tr>
  <tr>
    <td><p align="center">-</p></td>
    <td><p align="center"><a href="https://velog.io/@dlgkdis801">velog.io/@dlgkdis801</a></p></td>
    <td><p align="center">-</p></td>
  </tr>
  <tr>
    <td><p align="center">총괄, 기획</p></td>
    <td><p align="center">검증, 배포, 진행사항 정리</p></td>
    <td><p align="center">설계, 확장</p></td>
  </tr>
</table>



### 동시 테스트 진행 및 작업 수행
- RAG 시스템 특성으로 인해 작업을 동시에 수행 + 구조 내 특정 부분에 대한 테스트 수행 방식으로 작업

<br>
<br>

## DataSet
### KIS Weekly Report
- 🔗 [KIS자산평가/자료실/Weekly](https://www.bond.co.kr/post/10106/10254#none;)
- 발행 주관 : KIS 자산평가
- 발행 주기 : week
- 활용 범위 : 제1090호 - 제1120호 (2024년 6월 28일 - 2025년 1월 24일)
- 특징 : Chart, Table, Text가 혼합된 Multi Data 형태

<br>
<br>

## 개요
### 1. 다양한 매체를 통한 투자 정보 수집
- 실제 투자 상황에 활용할만한 신뢰도 부족
- 매일 쏟아지는 Report, 재무제표, 기사에 대한 분석 시간 및 비용 비효율 발생
#### ➡️ LLM(Large Language Model) 활용

<br>

### 2. 기존 LLM의 문제점
-  Perplexity, Tavily 등의 검색 기능이 탑재된 LLM을 활용하면 빠른 정보 수집 가능
-  그러나 다음의 문제점 발생
  - 정보 신뢰성
  - 사용자에게 fit한 Report인지 여부
  - Report 내용 정확성
  - 원하는 정보를 얻기 위한 적절한 LLM 활용 필요
#### ➡️ 채권 시장 리포트 수치 데이터의 효율적 요약 및 분석 대화형 챗봇 제작의 필요성

<br>

### 3. 안정성과 신뢰성을 고려한 Archtecture 구축 및 실험
- 확장 가능성 고려 : 추후 더 많은 Report가 들어오더라도 문제 없이 정보 제공
- Focus : 아키텍쳐에 집중
  - 잘못된 정보 전달 대신 -> "모른다"는 답변이 신뢰도에 도움이 될 것
  - 답변 출력 형식, 시스템 속도 문제보다 -> 신뢰도가 아주 중요
#### ➡️ 어떻게 하면 정확한 데이터만 제공하는 RAG 시스템을 구축할 수 있는가에 대한 부분이 주요 Task

<br>
<br>

## 프로젝트 수행
### 1. Parsing
- LangGraph 및 Upstage Layout Analyzer를 통한 요소별 데이터를 좌표 기준으로 추출
- Text
  - Rule-Based로 전처리
- Table
  - Llama Parser를 이용해 Markdown 형태로 추출
- Chart
  - OCR을 이용해 내용 추출

<br>

### 2. VectorDB
- Structure
  - [ATTU](https://github.com/zilliztech/attu) 활용
  - Text, Table, Raptor, Image로 구성
  - Index Type & Metric Type : AUTOINDEX - IP

<br>

### 3. 평가 방법
- AutoRAG를 이용한 QA Set 생성
- RAGAS 평가 지표 4가지 활용 : `Context Recall`, `Context Precision`, `Answer Similarity`, `Answer Correctness`

<br>

### 4. Embedding
- Upstage Solar Embedding 및 [bge-m3](https://huggingface.co/BAAI/bge-m3) 이용

<br>

### 5. Chunking
- Sementic Chunker 사용
  - parameter : Percentile 80% based breakpoint

<br>

### 6. 날짜 처리
- Milvus 메타데이터 필터링 기능 사용: User Query에서 정확한 날짜 범위 추출

<br>

### 7. Retriever
- Text : Milvus Retriever(textDB) + LLMReranking
- Table : Milvus Retriever(tableDB) + KiwiBM25Retriever(table)
- Raptor(date) : Milvus Retriever(raptorDB)
- Raptor(number) : Self Query Retriever(raptorDB)
- Image : Milvus Retriever(imageDB)

<br>

### 8. 문서 요약
- Raptor 방법론을 이용한 요약 Vector 생성 및 별도 Collection으로 저장하여 사용
- 논문 참고 : [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/html/2401.18059v1)
- 호수와 관련된 질문은 SelfQuery, 날짜와 관련된 질문은 일반 Milvus Retriever로 검색 수행

<br>

### 9. Query Routing


<br>
<br>

## 디렉터리 구조
```
├── 📑 README.md
├── 📑                         # 
├── 📑                         # 
├── 🗂️                         # 
|   ├──                        # 
|   └──                        # 
|
├── 📑                          # 
└── 📑                          # 
```

<br>
<br>

## 기술 스택
> ### Language
- Python `3.11`

<br>

> ### Format
- JSON
- JSONLines
- CSV

<br>

> ### API & Cloud
- OpenAI `1.61.1`
- Upstage `0.4.0`
- HuggingFace 
- GCP(Google Cloud Platform)

<br>

> ### RAG
- LangChain `0.3.19`
- RAGAS `0.1.21`
- AutoRAG `0.3.13`

<br>

> ### VectorDB & Embedding
- Milvus
  - langchain_milvus
  - langchain_community.vectorstores
- Embeddings
  - UpstageEmbeddings
  - OpenAIEmbeddings
  - HuggingFaceEmbeddings

<br>

> ### Library
- LangChain
  - community `0.3.18`
  - openai `0.2.14`
  - core `0.3.40`
  - experimental `0.3.4`
  - upstage `0.4.0`
  - milvus `0.1.8`

<br>
<br>

## 프로젝트 일정
![Image](https://github.com/user-attachments/assets/7c86ddc6-42c8-4983-8e81-5d5ac21a3953)

<br>
<br>

## 참고 문서 및 코드 참고
