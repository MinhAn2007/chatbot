import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})


def load_preprocessed_text():
    """Load preprocessed text instead of PDF"""
    loader = TextLoader("preprocessed_data.txt")
    text = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Reduced chunk size
        chunk_overlap=30,
        length_function=len
    )
    return text_splitter.split_documents(text)


def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",  # Smaller model
        google_api_key=API_KEY,
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})  # Reduced retrieval
    prompt = PromptTemplate.from_template(
        """Answer the question based strictly on the context. 
        If you can't find the answer, say "I don't know".
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


# Global variable for vectorstore
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
            chunks = load_preprocessed_text()
            VECTORSTORE = create_vectorstore(chunks)

        rag_chain = create_rag_chain(VECTORSTORE)
        response = rag_chain.invoke(question)
        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0')



