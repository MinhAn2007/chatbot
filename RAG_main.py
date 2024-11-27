import os
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


def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=API_KEY,
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    prompt = PromptTemplate.from_template(
        """Trả lời câu hỏi dựa trên nội dung đã cung cấp. 
        Nếu không tìm được câu trả lời, hãy nói: "Xin lỗi tôi không thể hỗ trợ bạn việc này, vui lòng liên hệ trực tiếp Zalo: 0837710747 hoặc Gmail: voongocminhan20072002@gmail.com để được hỗ trợ thêm."
        Câu hỏi: {question}
        Nội dung: {context}
        Câu trả lời:"""
    )
    return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )


# Global variable to store vectorstore
VECTORSTORE = None


@app.route('/ask', methods=['POST'])
def ask_question():
    global VECTORSTORE

    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "Không có câu hỏi được cung cấp"}), 400

    try:
        if VECTORSTORE is None:
            chunks = load_pdf_chunks()
            VECTORSTORE = create_vectorstore(chunks)

        rag_chain = create_rag_chain(VECTORSTORE)
        response = rag_chain.invoke(question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0')