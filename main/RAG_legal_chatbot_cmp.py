import os
import time
from tqdm import tqdm
from lib.Utils import GetRoot
from lib.Utils import GetHfToken
from lib.RAG import LlmWithRAG
from lib.RAG import LlmWithoutRAG


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
dir_embeddings = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "embeddings"
)
dict_embedding_json = {
    "embedding_1r_chunk_MiniLM": os.path.join(dir_embeddings, "laws_embedding_1chunk_0overlap_MiniLM.json"),
    "embedding_1r_chunk": os.path.join(dir_embeddings, "laws_embedding_1chunk_0overlap_Llama.json"),
    "embedding_3r_chunk": os.path.join(dir_embeddings, "laws_embedding_3chunk_1overlap_Llama.json"),
    "embedding_4r_chunk": os.path.join(dir_embeddings, "laws_embedding_4chunk_2overlap_Llama.json")
}

# Create result saving directory
dir_qa_results = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_RESULTS"),
    "qa_results"
)
os.makedirs(dir_qa_results, exist_ok=True)


########## Start Processing ##########
# 紀錄執行時間
time_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# Models
lis_llm = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
]

# Obtain the queries (6/10)
lis_queries = []
with open(fil_query_path, "r", encoding="utf-8") as f:
    for line in f:
        if "Case" in line:
            query = line.strip("\n").split(": ")[-1]
            lis_queries.append(query)
lis_queries = lis_queries[4:]

# Loop the models, embedding files, and the query file
max_output_token = 512
str_desc = "問答進行中流程..."
for llm in tqdm(lis_llm, desc=str_desc):
    llm_name = llm.split("/")[-1]
    print(f">>> 使用模型: {llm_name}")

    lis_record = []
    query_num = 0
    for query in lis_queries:
        print(">>> 沒有使用RAG處理")
        query_num += 1
        query_record = f"==========\n問題 {query_num}: " + query.strip("\n") + "\n=========="
        lis_record.append(query_record)

        # 沒有使用RAG流程
        answer_no_rag = LlmWithoutRAG(
            llm_model=llm,
            query=query,
            max_token=max_output_token,
        )
        str_model = f"模型: {llm}"
        lis_record.append(str_model)
        str_without_rag = "沒有使用RAG處理"
        lis_record.append(str_without_rag)
        lis_record.append(answer_no_rag)
        lis_record.append("==========")

        # 使用RAG流程
        for dataset_name, dataset_path in dict_embedding_json.items():
            print(f">>> 使用詞向量資料庫: {dataset_name}")
            if dataset_name == "embedding_1r_chunk_MiniLM":
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                embedding_model = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"

            answer_with_rag = LlmWithRAG(
                embedding_model=embedding_model,
                embedding_dataset=dataset_path,
                llm_model=llm,
                max_token=max_output_token,
                query=query,
                top_k=5
            )
            str_model = f"模型: {llm}"
            lis_record.append(str_model)
            str_embedding_dataset = f"使用的詞向量資料庫: {dataset_name}"
            lis_record.append(str_embedding_dataset)
            lis_record.append(answer_with_rag)
            lis_record.append("==========")

    llm_name = llm.split("/")[-1].replace('.', '').replace('-', '_')
    fil_result = os.path.join(dir_qa_results, f"answers_{llm_name}_{time_tag}.txt")
    str_desc = "資料儲存中..."
    with open(fil_result, 'w', encoding="utf-8") as f:
        for line in tqdm(lis_record, desc=str_desc):
            line += "\n"
            f.write(line)
    print(f"處裡完成，結果儲存於: {fil_result}")
