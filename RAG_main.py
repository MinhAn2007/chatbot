import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Set API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key not found in environment variables")

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})


def load_and_split_pdf(pdf_url):
    try:
        print(f"Loading PDF from: {pdf_url}")
        loader = PyPDFLoader(pdf_url)
        docs = loader.load()
        if not docs:
            raise ValueError("No documents loaded from PDF")

        print(f"Successfully loaded {len(docs)} pages from PDF")

        # Print content of the first page to debug
        for i, doc in enumerate(docs):
            print(f"\nPage {i + 1} content preview:")
            print(doc.page_content[:500])  # Print first 500 chars of each page
            print(f"Page {i + 1} total length: {len(doc.page_content)} characters")

        # Create text splitter with smaller chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced from 1024
            chunk_overlap=50,  # Reduced from 100
            length_function=len,
            is_separator_regex=False
        )

        # Split documents and add debugging
        chunks = text_splitter.split_documents(docs)
        print(f"\nSplit documents into {len(chunks)} chunks")

        if chunks:
            print("\nFirst chunk preview:")
            print(chunks[0].page_content[:200])
            print(f"First chunk length: {len(chunks[0].page_content)} characters")
        else:
            print("\nDocument content appears to be empty or unsplittable")
            # If no chunks were created, try using the entire document as one chunk
            if docs[0].page_content.strip():
                print("Using entire document as a single chunk")
                return docs

        if not chunks:
            raise ValueError("No chunks created from documents")

        return chunks
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        raise


def initialize_vectorstore(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Filter out complex metadata that might cause issues
        filtered_chunks = filter_complex_metadata(chunks)

        if not filtered_chunks:
            raise ValueError("No valid chunks after filtering metadata")

        print(f"Creating vector store with {len(filtered_chunks)} chunks")

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embeddings
        )

        return vectorstore
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise


def setup_rag_chain():
    try:
        # Load and process PDF
        chunks = load_and_split_pdf("https://fashionandl.s3.ap-southeast-1.amazonaws.com/zalo/data.pdf")

        # Initialize vector store
        vectorstore = initialize_vectorstore(chunks)

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=api_key,
            temperature=0,
            streaming=True
        )

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 3}
        )

        # Define prompt
        prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Answer the question based on your own knowledge.
            If you don't know the answer, just say that you don't know. Keep your answer straightforward and concise. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

        # Create RAG chain
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        return rag_chain
    except Exception as e:
        print(f"Error setting up RAG chain: {str(e)}")
        raise


# Initialize RAG chain
try:
    rag_chain = setup_rag_chain()
except Exception as e:
    print(f"Failed to initialize RAG chain: {str(e)}")
    raise


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Invoke RAG chain
        response = rag_chain.invoke(question)
        return jsonify({"question": question, "answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/hello")
def hello():
    return {"message": "Test ok!!"}


def lambda_handler(event, context):
    query = event.get("question")
    if not query:
        return {"statusCode": 400, "body": "No question provided"}

    try:
        response = rag_chain.invoke(query)
        return {"statusCode": 200, "body": {"question": query, "answer": response}}
    except Exception as e:
        return {"statusCode": 500, "body": str(e)}


if __name__ == '__main__':
    app.run(host='0.0.0.0')