# 🕵️ Bogosa
### [AIFFELthon DS 3기] 보고사: 채권 리포트 기반 VectorDB 구축 및 RAG 최적화
- 금융 보고서 기반 최적 VectorDB 생성 & RAG 기법 최적화 및 신뢰도 높은 금융 특화 LLM 환경 구축 Project
- [NotionPage/보고사](https://hayanlee.notion.site/1c4022a887d980658b1cfafc1f23936a?pvs=4)

![Image](https://github.com/user-attachments/assets/f77f4762-4e6f-47ee-ad62-d60a2dc08621)

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

## 디렉터리 구조
```
├── 📑 README.md
├── 📑 requirements.txt                  # 설치 파일
|
├── 🗂️ docs                              # 프로젝트 로그 및 전반 내용 작성
|   ├── 📑 아이펠톤_최종발표자료_보고사.pdf
|   ├── 📑 프로젝트 계획서
|   └── 🔗 NotionPage/보고사
|
├── 🗂️ bogosa                            
|   ├── 📑 250321_LCEL.ipynb             # 메인 파일
|   ├── 🗂️ QA                            # QA셋
|   ├── 🗂️ raptor                        # raptor 알고리즘 적용 jsonl 파일
|   └── 🗂️ chunked_jsonl                 # 청킹 완료된 jsonl 파일
|
├── 🗂️ evaluation
|   ├── 🗂️ AutoRAG                       # AutoRAG 활용 QA 데이터셋 생성 실행 파일
|   └── 🗂️ RAGAS                         # RAGAS 모듈 관련 정리
|       ├── 🗂️ score                     # DB별 성능 지표
|       └── 🗂️ test_notebook             # 성능 평가 ipynb 파일
|
├── 🗂️ streamlit                         # streamlit 관련 정리
|   ├── 📑 streamlit.py                  # UI
|   └── 📑 financial_qa.py               # Code
|
└── 🗂️ pipeline                          # 파이프라인
     ├── 🗂️ text                         # 텍스트 파이프라인
     └── 🗂️ Airflow                      # Airflow DAG
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
- LangGraph `0.2.74`
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
- 질문 유형에 따른 최적화된 답변 제공을 위해 사용
- 질문 입력 ➡️ 날짜 관련 질문, 특정 호수 질문, 범용 질문인지 판단
  - 날짜 관련 질문 : 요약, 예측, 일반 질문인지를 추가로 구분
- 각 질문에 최적 프롬프트 및 체인을 사용하도록 구성

<br>

### 10. Chart
- 이미지에서 직접적 정보를 추출하는 것은 위험성 존재 ➡️ 참고 자료로 활용
- RAG 체인을 통해 얻은 답변 + image vectorstore에서 검색된 이미지 요약 ➡️ Groundedness Checker로 검증

<br>
<br>


## RAG Architecture
![Image](https://github.com/user-attachments/assets/6c146da3-ec08-479f-b328-05001c41b698)
![Image](https://github.com/user-attachments/assets/ea56c403-18ae-405d-aa5c-f2467d3b3a58)

<br>
<br>

## 검색 수행 결과
> ### 📍 시연 영상
> [![Video Label](http://img.youtube.com/vi/kbU95wzck3s/0.jpg)](https://youtu.be/kbU95wzck3s)

<br>

### 💬 "2주전 은행채 발행액은?" <br>
![Image](https://github.com/user-attachments/assets/2fb82ddf-7ffb-4205-9d47-b8cb78391b42)

<br>

### 💬 "2025년 1월 24일과 2025년 1월 17일의 국공채 시장 동향을 비교하시오." <br>
![Image](https://github.com/user-attachments/assets/0488645a-7bf5-4d22-ba0c-ca64e2f8510d)

<br>

### 💬 "지난 달 은행채 발행액 총액은?" <br>
![Image](https://github.com/user-attachments/assets/32eb0665-5a87-4e50-a60a-f1082805b160)

- 실제 12월 1 - 4주차 은행채 발행액 정보 <br>
![Image](https://github.com/user-attachments/assets/874940de-357f-4ea1-895f-72aa83bdc4bf)

<br>

### "💬 1101호 요약해줘." <br>
![Image](https://github.com/user-attachments/assets/81d0e217-7f84-4de7-bb3d-a016313330af)

<br>

### 💬 "다음달 회사채 시장과 스프레드 전망 알려줘." <br>
![Image](https://github.com/user-attachments/assets/24df717f-cda2-48fd-a7fa-816d48031316)

<br>
<br>

## 참고 문서 및 코드 참고
- 참고
  - [<랭체인LangChain 노트> - LangChain 한국어 튜토리얼🇰🇷](https://wikidocs.net/233341)
  - [이토록 쉬운 RAG 시스템 구축을 위한 랭체인 실전 가이드](https://www.yes24.com/product/goods/136548871)
  - [AutoRAG 튜토리얼](https://www.youtube.com/playlist?list=PLIMb_GuNnFwdjfLUPrpUAzjQLfBJLQ7MC)
  - [LoRA의 개념](https://www.youtube.com/watch?v=0lf3CUlUQtA)
  - [BigQuery RAG 파이프라인(Document AI Layout Parser)](https://cloud.google.com/blog/ko/products/data-analytics/bigquery-and-document-ai-layout-parser-for-document-preprocessing)

- Milvus
  - [milvus 메타데이터 필터링](https://milvus.io/docs/ko/filtered-search.md)
  - [Efficiently Deploying Milvus on GCP Kubernetes: A Guide to Open Source Database Management](https://medium.com/@zilliz_learn/efficiently-deploying-milvus-on-gcp-kubernetes-a-guide-to-open-source-database-management-7e49d0b194d8)
  - [Command line tool (kubectl)](https://kubernetes.io/docs/reference/kubectl/)
  - [Kubernetes CLI 도구인 kubectl의 사용법 이해하기](https://velog.io/@pinion7/kubernetes-CLI-%EB%8F%84%EA%B5%AC%EC%9D%B8-kubectl%EC%9D%98-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)
  - [GKE에 Milvus 클러스터 배포하기](https://milvus.io/docs/ko/gcp.md)

<br>

- 논문
  - [AutoRAG를 이용한 금융 문서에 가장 최적화된 RAG 시스템 구현에 관한 연구](https://koreascience.or.kr/article/CFKO202433162114304.pdf)
  - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)
  - [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/html/2401.18059v1)

<br>

- 공식문서
  - [AutoRAG 평가 지표](https://docs.auto-rag.com/evaluate_metrics/retrieval.html)

<br>

- 사례
  - [신한투자증권 X 스켈터랩스 :: 증권사 RAG 활용 사례](https://www.skelterlabs.com/blog/rag-securities)
  - [기업용 금융 특화 LLM 모델 만들기 (1)- 필요성과 RAG](https://blog-ko.allganize.ai/alli-finance-llm-1/)

<br>

- Model
  - [mteb_ko_leaderboard(오픈소스 임베딩 모델)](https://github.com/su-park/mteb_ko_leaderboard)
