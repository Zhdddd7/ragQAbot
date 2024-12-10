from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 更新导入路径
from langchain_community.vectorstores import FAISS

# 加载 PDF 文件
pdf_loader = PyPDFLoader('nlptextbook.pdf', extract_images=False)

# 分块处理文本
chunks = pdf_loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
)

# 加载 Hugging Face 的 embedding 模型
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 创建 FAISS 向量库
vector_db = FAISS.from_documents(chunks, embedding_model)
vector_db.save_local('LLM.faiss')

print("FAISS saved!")
