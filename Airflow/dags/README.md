# Airflow 파이프라인
- Airflow 및 GitHub Actions를 이용한 RAG 시스템 PipeLine 구축 (CI)

<br>

## 1. Airflow
![Image](https://github.com/user-attachments/assets/a5e20fd3-125b-4e34-a326-c03bed2c4142)
- 현재의 RAG 시스템을 파이프라인으로 올리기 위한 용도
- 새로운 PDF를 적용할 경우, 파이프라인을 통해 바로 시스템에 적용하기 위한 목적

### DAG 생성 테스트
- DAG 생성 테스트
  - 필수 요소
    - Airflow 설치 (pip install apache-airflow)
    - ~/Airflow 폴더 내 dags 폴더 내부 ➡️ .py로 저장 시 DAG 생성됨
      - 주의 : 모든 새로운 코드는 dags 폴더 내에 저장되어야 import 가능

- Skeleton 코드에 Airflow 적용
  - PDF ➡️  split, embedding, vectorStore 적재까지 진행되도록 구성하여 테스트 완료
  - PDF Processing 후 XCom에 임시 저장하여 다음 task로 전달하는 방식 사용

- In progress
  - 현재 RAG 시스템 함수화 & DAG task 생성
  - CI 스크립트 설계
  - GitHub Actions -> Airflow의 파이프라인 자동 감지
