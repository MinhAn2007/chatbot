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

# Set OpenAI API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize  
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}, r"/ask": {"origins": "*"}})

llm_model = "gemini-1.5-flash-latest"  # Set this to "gemini-1.5" once available

# Initialize LangChain components
llm = ChatGoogleGenerativeAI(api_key=api_key, model=llm_model, temperature=0, streaming=True)

# Define a prompt to be used by the LLM
prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved
            context to answer the question.   
 If you don't know the answer, just say that you don't know.   

            Keep your answer straight forward and concise. No yapping! [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
)

prompt2 = PromptTemplate.from_template(
    """
     <s> [INST] You are an assistant for question-answering tasks. Answer the question based on your own knowledge.
     If you don't know the answer, just say that you don't know. Keep your answer straightforward and concise. No yapping! [/INST] </s>
     [INST] Question: {question}
     Context: {context}
     Answer: [/INST]
     """
)

# Load a PDF file
docs = PyPDFLoader(file_path="document/height_weight_shirt_size_table.pdf").load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=1024,
       chunk_overlap=100
)

# Split doc into smaller chunks of text
chunks = text_splitter.split_documents(docs)

chunks = filter_complex_metadata(chunks)

vectorstore = Chroma.from_documents(documents=chunks, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(

            search_type='similarity',
            search_kwargs={
                'k': 3,
            },
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt2
    | llm
    | StrOutputParser()
)


def ask_question(question):
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Invoke RAG chain with the provided question
    response = rag_chain.invoke(question)

    return jsonify({"question": question, "answer": response})

def lambda_handler(event, context):
    query = event.get("question")
    response = ask_question(query)
    print("response:", response)
    return {"body": response, "statusCode": 200}

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Invoke RAG chain with the provided question
    response = rag_chain.invoke(question)

    return jsonify({"question": question, "answer": response})

@app.get("/hello")
async def hello():
    return  {"message": "Test ok!!"}


if __name__ == '__main__':
    app.run(host='0.0.0.0')

    def handler(event, context):
        return app