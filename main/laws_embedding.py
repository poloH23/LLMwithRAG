import os
import json
import pathlib
from tqdm import tqdm
from lib.Utils import GetRoot
from lib.Embedding import GetHfToken
from lib.Embedding import EmbeddingByMiniLM
from lib.Embedding import EmbeddingByLlama
from sentence_transformers import SentenceTransformer


# Load environment variables
GetRoot()

# Create result saving directory
dir_embeddings = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "embeddings"
)
os.makedirs(dir_embeddings, exist_ok=True)

# Obtain the input data
fil_laws_path = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "laws",
    "laws_and_content.txt"
)

# Add HuggingFace token
token_info = GetHfToken()
print(token_info) if token_info is not None else print(">>> HuggingFace token NOT found.")


"""
########## Condition 1 ##########
# Embedding model: sentence-transformers/all-MiniLM-L6-v2
# chunk: 每一條法條
# overlap: 無
# output file: laws_embedding_1chunk_0overlap_MiniLM.json
fir_emb_minilm = os.path.join(
    dir_embeddings,
    "laws_embedding_1chunk_0overlap_MiniLM.json"
)

# 轉換法條文本為詞向量
model_name = "sentence-transformers/all-MiniLM-L6-v2"
EmbeddingByMiniLM(
    input_path=fil_laws_path,
    output_path=fir_emb_minilm,
    model_name=model_name,
    batch_size=32
)
"""

"""
########## Condition 2 ##########
# Embedding model: lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct
# chunk: 每一條法條
# overlap: 無
# output file: laws_embedding_1chunk_0overlap.json
fir_emb_llama_1c0o = os.path.join(
    dir_embeddings,
    "laws_embedding_1chunk_0overlap_Llama.json"
)

# 轉換法條文本為詞向量
model_name = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
EmbeddingByLlama(
    input_path = fil_laws_path,
    output_path = fir_emb_llama_1c0o,
    model_name = model_name,
    batch_size = 32
)
"""

