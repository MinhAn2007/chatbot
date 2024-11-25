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
TIMEOUT_SECONDS = 30  # Timeout sau 30 giây


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
    response = chain.invoke(question)

    # Nếu response trống hoặc quá ngắn, trả về message mặc định
    if not response or len(response.strip()) < 10:
        return DEFAULT_ERROR_MESSAGE

    # Kiểm tra thời gian xử lý
    if time.time() - start_time > TIMEOUT_SECONDS:
        return DEFAULT_ERROR_MESSAGE

    return response


def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=API_KEY,
        temperature=0,
        max_output_tokens=1024,  # Giới hạn độ dài output
        timeout=TIMEOUT_SECONDS  # Timeout cho LLM
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            'k': 3,
            'fetch_k': 5,
            'score_threshold': 0.5  # Thêm ngưỡng điểm để lọc kết quả không liên quan
        }
    )

    def check_relevance(retriever_output):
        documents = retriever_output['documents']

        if not documents:
            return DEFAULT_ERROR_MESSAGE

        # Kiểm tra độ liên quan của documents
        total_score = sum(doc.metadata.get('score', 0) for doc in documents)
        avg_score = total_score / len(documents) if documents else 0

        if avg_score < 0.5:  # Ngưỡng điểm trung bình
            return DEFAULT_ERROR_MESSAGE

        return retriever_output['context']

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant that answers questions based on the given context.
        If you cannot find a relevant answer in the context, respond with exactly: "{error_message}"
        Do not try to generate answers from outside the context.

        Question: {question}
        Context: {context}

        Answer (Be concise and direct):""".format(
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
            chunks = load_pdf_chunks()
            VECTORSTORE = create_vectorstore(chunks)
            RAG_CHAIN = create_rag_chain(VECTORSTORE)

        # Process with timeout
        response = process_rag_query(RAG_CHAIN, question)

        # Kiểm tra response
        if not response or response.isspace() or response == DEFAULT_ERROR_MESSAGE:
            return jsonify({"answer": DEFAULT_ERROR_MESSAGE}), 200

        return jsonify({"answer": response}), 200

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"answer": DEFAULT_ERROR_MESSAGE}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')