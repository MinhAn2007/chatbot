import os
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})

DEFAULT_ERROR_MESSAGE = "Xin lỗi tôi không thể hỗ trợ bạn việc này, xin vui lòng liên hệ trực tiếp zalo: 0837710747 hoặc gmail: voongocminhan20072002@gmail.com để biết thêm chi tiết"
TIMEOUT_SECONDS = 30


def timeout_handler(timeout):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError:
                    return DEFAULT_ERROR_MESSAGE

        return wrapper

    return decorator


def load_pdf_chunks():
    loader = PyPDFLoader("https://fashionandl.s3.ap-southeast-1.amazonaws.com/zalo/data.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.split_documents(docs)


def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


@timeout_handler(TIMEOUT_SECONDS)
def process_rag_query(chain, question):
    start_time = time.time()
    try:
        response = chain.invoke(question)

        # Kiểm tra response
        if not response or len(response.strip()) < 10:
            return DEFAULT_ERROR_MESSAGE

        # Kiểm tra thời gian xử lý
        if time.time() - start_time > TIMEOUT_SECONDS:
            return DEFAULT_ERROR_MESSAGE

        return response
    except Exception as e:
        print(f"Error in process_rag_query: {str(e)}")
        return DEFAULT_ERROR_MESSAGE


def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=API_KEY,
        temperature=0,
        max_output_tokens=1024,
        timeout=TIMEOUT_SECONDS
    )

    # Cấu hình retriever phù hợp với Chroma
    retriever = vectorstore.as_retriever(
        search_kwargs={
            'k': 5,  # Tăng số lượng documents để có nhiều context hơn
            'score_threshold': 0.5
        }
    )

    def check_relevance(retriever_output):
        if isinstance(retriever_output, str):
            return retriever_output

        documents = retriever_output.get('documents', [])
        if not documents:
            return DEFAULT_ERROR_MESSAGE

        # Lọc documents có điểm số thấp
        relevant_docs = []
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata.get('score', 0) >= 0.5:
                relevant_docs.append(doc)

        if not relevant_docs:
            return DEFAULT_ERROR_MESSAGE

        # Kết hợp nội dung của các documents liên quan
        combined_context = "\n".join(doc.page_content for doc in relevant_docs)
        return combined_context

    prompt = PromptTemplate.from_template(
        """Bạn là trợ lý AI giúp trả lời câu hỏi dựa trên context cho trước.
        Hãy trả lời câu hỏi dựa CHÍNH XÁC trên context được cung cấp.
        Nếu không tìm thấy thông tin liên quan trong context, hãy trả lời chính xác message sau: "{error_message}"
        KHÔNG được tự tạo câu trả lời ngoài context.

        Question: {question}
        Context: {context}

        Trả lời ngắn gọn và chính xác:""".format(
            error_message=DEFAULT_ERROR_MESSAGE,
            question="{question}",
            context="{context}"
        )
    )

    return (
            {
                "context": retriever | check_relevance,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )


# Global variables
VECTORSTORE = None
RAG_CHAIN = None


@app.route('/ask', methods=['POST'])
def ask_question():
    global VECTORSTORE, RAG_CHAIN

    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Initialize once
        if VECTORSTORE is None:
            print("Initializing vectorstore and RAG chain...")
            chunks = load_pdf_chunks()
            VECTORSTORE = create_vectorstore(chunks)
            RAG_CHAIN = create_rag_chain(VECTORSTORE)
            print("Initialization complete")

        # Process with timeout
        start_time = time.time()
        response = process_rag_query(RAG_CHAIN, question)
        process_time = time.time() - start_time
        print(f"Query processed in {process_time:.2f} seconds")

        # Kiểm tra response
        if not response or response.isspace() or response == DEFAULT_ERROR_MESSAGE:
            return jsonify({"answer": DEFAULT_ERROR_MESSAGE}), 200

        return jsonify({"answer": response}), 200

    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return jsonify({"answer": DEFAULT_ERROR_MESSAGE}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')