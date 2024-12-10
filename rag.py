from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import RetrievalQA
import torch
# Load Embedding and Tokenizer
model_name = "meta-llama/Llama-3.2-3B"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(model.hf_device_map)
# breakpoint()
# Create HuggingFacePipeline


pipeline_llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,  
)


langchain_llm = HuggingFacePipeline(pipeline=pipeline_llm)

# Vectorization
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load Vector database for vector recall
print("Loading FAISS vector database...")
vector_db = FAISS.load_local('LLM.faiss', embedding_model, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Build retriveval for RAG paradigm
retrieval_qa = RetrievalQA.from_chain_type(
    llm=langchain_llm,
    chain_type="stuff",
    retriever=retriever
)

# Chat
chat_history = []
while True:
    query = input("query: ")
    # RAG
    print("Processing query...")
    result = retrieval_qa.invoke({"query": query})
    response_content = result["result"]
    print('-------------------------------------')
    print(response_content)
    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response_content))
    chat_history = chat_history[-20:]  # 保留最近 10 轮对话m
