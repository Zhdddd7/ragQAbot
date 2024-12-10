from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 更新导入路径
from langchain_community.vectorstores import FAISS

# Load PDF
pdf_loader = PyPDFLoader('nlptextbook.pdf', extract_images=False)

# file chunking
chunks = pdf_loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
)

# Load huggingface model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Creat FAISS vector database
vector_db = FAISS.from_documents(chunks, embedding_model)
vector_db.save_local('LLM.faiss')

print("FAISS saved!")
