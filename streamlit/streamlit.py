import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

from financial_qa import FinancialQASystem, EmbeddingManager, VectorStoreManager, TimeProcessor, DocumentProcessor, Reranker, PromptManager, ChainManager, Router
from langchain_core.messages.chat import ChatMessage


import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')


class StreamlitFinancialQA:
    def __init__(self):
        self.qa_system = FinancialQASystem()

    def run(self):
        st.title("금융 Q&A 시스템")

        # 처음 1번만 실행하기 위한 코드
        if "messages" not in st.session_state:
            # 대화기록을 저장하기 위한 용도로 생성한다.
            st.session_state["messages"] = []

        # 사이드바 생성
        with st.sidebar:
            # 초기화 버튼 생성
            clear_btn = st.button("대화 초기화")

        user_input = st.chat_input("궁금한 내용을 물어보세요!")

        if user_input:
            st.chat_message("user").write(user_input)

            response = self.qa_system.ask(user_input)
            images = self.display_images(user_input)
            with st.chat_message("assistant"):
                container = st.empty()

                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)
                if len(images) != 0:
                  with st.expander('참고 자료'):
                    st.image(images)
                      
            self.add_message("user", user_input)
            self.add_message("assistant", ai_answer)

    def display_images(self, question):
        expr = self.qa_system.time_processor.get_query_date(question)
        context = self.qa_system.vector_store_manager.get_retriever('image', {'k': 3}).invoke(question, expr=expr)

        img = []
        for i in context:
            rar = self.qa_system.chain_manager.query_retrieval_relevant.invoke({'context': i, 'question': question})
            if rar.score == 'yes':
                image_path = i.metadata['image'].replace('raw_pdf_copy3', 'parsed_pdf')
                img.append(Image.open(image_path))
        return img
    
    # 이전 대화를 출력
    def print_messages(self):
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)


    def add_message(self, role, message):
        st.session_state["messages"].append(ChatMessage(role=role, content=message))


if __name__ == "__main__":
    app = StreamlitFinancialQA()
    app.run()