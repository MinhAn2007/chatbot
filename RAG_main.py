import os
import requests
import json
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
from langchain_core.documents import Document

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})


def fetch_product_data():
    """
    Fetch product data from the specified API and convert to a single document chunk.
    """
    try:
        print("Attempting to fetch products...")

        response = requests.get("https://bestore-murex.vercel.app/api/products", timeout=10)
        response.raise_for_status()

        response_json = response.json()
        products = response_json.get('products', [])
        print(f"Number of products found: {len(products)}")

        product_text = ""
        for product in products:
            # Dùng set để loại bỏ trùng lặp
            price_list = [float(sku['price']) for sku in product.get('skus', []) if 'price' in sku]
            colors = list(
                set(sku['color'] for sku in product.get('skus', []) if 'color' in sku))  # Loại bỏ màu trùng lặp
            sizes = list(
                set(sku['size'] for sku in product.get('skus', []) if 'size' in sku))  # Loại bỏ kích thước trùng lặp

            product_text += f"""
            Thông tin sản phẩm:
            ID Sản phẩm: {product.get('id', 'Không có thông tin')}
            Tên sản phẩm: {product.get('name', 'Không có thông tin')}
            Các mức giá: {str(price_list)}
            Các màu sắc: {str(colors)}
            Các Kích thước: {str(sizes)}
            Đã bán: {product.get('sold', 'Không có thông tin')}
            Link mua sản phẩm: https://fashion-store-rouge.vercel.app/{product.get('id', '')}
            \n"""  # Thêm ký tự xuống dòng để phân cách các sản phẩm

        # Tạo document chứa toàn bộ dữ liệu sản phẩm
        product_doc = Document(
            page_content=product_text,
            metadata={
                'source': 'product_api',
                'product_count': len(products)
            }
        )

        print(f"Created product document with {len(products)} products")
        return [product_doc]
    except Exception as e:
        print(f"Error fetching product data: {e}")
        return []



def load_pdf_chunks():
    """
    Load and split PDF document into chunks.
    """
    try:
        loader = PyPDFLoader("https://fashionandl.s3.ap-southeast-1.amazonaws.com/zalo/data.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Loaded {len(chunks)} PDF chunks")
        return chunks
    except Exception as e:
        print(f"Error loading PDF: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_vectorstore(chunks):
    """
    Create a vector store from document chunks.
    """
    try:
        if not chunks:
            print("No chunks provided to create vectorstore")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY
        )
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
        print(f"Created vectorstore with {len(chunks)} chunks")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None


def create_rag_chain(vectorstore):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    """
    try:
        if vectorstore is None:
            print("Cannot create RAG chain: vectorstore is None")
            return None

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=API_KEY,
            temperature=0
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
        prompt = PromptTemplate.from_template(
            """Nhiệm vụ: Xử lý câu hỏi một cách thông minh và chính xác

            Quy tắc xử lý:
            1. Nếu là lời chào:
               - Trả lời bằng lời chào thân thiện, ấm áp
               - Sẵn sàng hỗ trợ người dùng

            2. Nếu là câu hỏi về sản phẩm:
               - Cung cấp thông tin chi tiết:
                 * Tên sản phẩm
                 * Mô tả sản phẩm
                 * Màu sắc có sẵn
                 * Kích thước có sẵn
                 * Đường dẫn mua hàng

            3. Xử lý khi không tìm thấy thông tin:
               - Trả lời: "Không tìm thấy sản phẩm phù hợp"

            4. Nguyên tắc chung:
               - Trả lời bằng tiếng Việt
               - Súc tích, dễ hiểu
               - Không giải thích thêm
               - Ưu tiên sử dụng thông tin từ nội dung được cung cấp
               - Nếu không tìm thấy thông tin trong nội dung:
                  * Sử dụng kiến thức chung để trả lời
                  * Bổ sung thông tin từ hiểu biết của bạn
               - Luôn đảm bảo câu trả lời chính xác, đầy đủ

            Câu hỏi: {question}
            Nội dung: {context}
            Câu trả lời:"""
        )

        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        print("RAG chain created successfully")
        return rag_chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        import traceback
        traceback.print_exc()
        return None


# Global variable to store vectorstore
VECTORSTORE = None
RAG_CHAIN = None


@app.route('/fetch_products', methods=['GET'])
def fetch_products_route():
    """
    Endpoint to directly fetch and display product data for debugging
    """
    try:
        products = fetch_product_data()
        return jsonify({
            "total_products": len(products),
            "products": [
                {
                    "product_id": doc.metadata.get('product_id', 'N/A'),
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                } for doc in products
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/debug_products', methods=['GET'])
def debug_products():
    """
    Endpoint to debug product fetching
    """
    products = fetch_product_data()
    return jsonify({
        "total_products": len(products),
        "products": [
            {
                "product_id": doc.metadata.get('product_id', 'N/A'),
                "content": doc.page_content
            } for doc in products
        ]
    })


@app.route('/ask', methods=['POST'])
def ask_question():
    global VECTORSTORE, RAG_CHAIN

    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "Không có câu hỏi được cung cấp"}), 400

    try:
        # Ensure we have a vectorstore
        if VECTORSTORE is None:
            # Load PDF chunks
            pdf_chunks = load_pdf_chunks()

            # Fetch product data and convert to documents
            product_chunks = fetch_product_data()

            # Combine PDF and product chunks
            all_chunks = pdf_chunks + product_chunks

            # Create vector store with combined chunks
            if not all_chunks:
                print("No chunks available to create vectorstore")
                return jsonify({"error": "Không thể tải dữ liệu"}), 500

            VECTORSTORE = create_vectorstore(all_chunks)

        # Ensure we have a RAG chain
        if RAG_CHAIN is None and VECTORSTORE is not None:
            RAG_CHAIN = create_rag_chain(VECTORSTORE)

        # Check if RAG chain is ready
        if RAG_CHAIN is None:
            print("Failed to create RAG chain")
            return jsonify({"error": "Không thể tạo chuỗi trả lời"}), 500

        # Invoke the RAG chain
        response = RAG_CHAIN.invoke(question)
        return jsonify({"answer": response})
    except Exception as e:
        print(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['GET'])
def reset_system():
    """
    Endpoint to reset the global vectorstore and RAG chain
    """
    global VECTORSTORE, RAG_CHAIN
    VECTORSTORE = None
    RAG_CHAIN = None
    return jsonify({"status": "System reset successfully"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)