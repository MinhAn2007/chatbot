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
        """Answer the question based strictly on the context. 
        If you can't find the answer, say "Xin lỗi tôi không thể hỗ trợ bạn việc này , xin vui lòng liên hệ trực tiếp zalo : 0837710747 hoặc gmail : voongocminhan20072002@gmail.com để biết thêm chi tiết"
        Question: {question}
        Context: {context}
        Answer:"""
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
        return jsonify({"error": "No question provided"}), 400

    try:
        # Create vectorstore only once
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