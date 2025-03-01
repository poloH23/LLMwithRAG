import os
import time
import random
from tqdm import tqdm
from lib.Utils import GetRoot
from lib.Utils import GetHfToken
from lib.RAG import LlmWithRAG
from lib.RAG import LlmWithoutRAG
from lib.RAG import LoadEmbeddings
from lib.RAG import CreateFaissIdx
from lib.RAG import SearchFaissIdx


# Load environment variables
GetRoot()

# Add HuggingFace token
token_info = GetHfToken()
print(token_info) if token_info is not None else print(">>> HuggingFace token NOT found.")

# Obtain the queries
fil_query_path = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "qa_file",
    "queries.txt"
)

# Obtain the embedding files
fil_embeddings = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "embeddings",
    "laws_embedding_4chunk_2overlap_Llama.json"
)

# Read the queries and randomly pick one question
lis_queries = []
with open(fil_query_path, "r", encoding="utf-8") as f:
    for line in f:
        if "Case" in line:
            query = line.strip("\n").split(": ")[-1]
            lis_queries.append(query)
str_query = random.choice(lis_queries)

# Define LLM model
str_model_name = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"

# Load the embeddings
texts, embeddings = LoadEmbeddings(embedding_path=fil_embeddings)

# Create FAISS indexing
index = CreateFaissIdx(embeddings=embeddings)

# Search related contents
top_results = SearchFaissIdx(
    model_name=str_model_name,
    query=str_query,
    idx=index,
    texts=texts,
    top_k=5,
    max_length=512
)

# Combine the researching results
context = "\n".join([result["text"] for result in top_results])

# Generate the response without RAG system application
none_rag = ''
answer_no_rag = LlmWithoutRAG(
    llm_model=str_model_name,
    query=str_query,
    max_token=512,
)

# Generate the response with RAG system application
answer_with_rag = LlmWithRAG(
    embedding_model=str_model_name,
    embedding_dataset=fil_embeddings,
    llm_model=str_model_name,
    max_token=512,
    query=str_query,
    top_k=5
)

# Print the response from LLM
print("========== Query ==========")
print(str_query)
print("========== RAG處理之前 ==========")
print(f">>> 使用模型: {str_model_name.split('/')[-1]}")
print(f">>> 回答: {answer_no_rag}\n\n")

print("========== RAG處理之後 ==========")
print(f">>> 使用模型: {str_model_name.split('/')[-1]}")
print(f">>> 使用詞向量: {fil_embeddings.split('/')[-1]}")
print(f">>> 回答: {answer_with_rag}")
