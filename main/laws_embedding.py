import os
from lib.Utils import GetRoot
from lib.Utils import GetHfToken
from lib.Embedding import EmbeddingByMiniLM
from lib.Embedding import EmbeddingByLlama
from lib.Embedding import ProcessLawText
from lib.Embedding import GenerateChunksWithOverlap
from lib.Embedding import ChunkEmbeddingByLlama
from lib.Embedding import GenerateChunksWithOverlapChar


# Load environment variables
GetRoot()

# Add HuggingFace token
token_info = GetHfToken()
print(token_info) if token_info is not None else print(">>> HuggingFace token NOT found.")

# Obtain the input data
fil_laws_path = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "laws",
    "laws_and_content.txt"
)

# Create result saving directory
dir_embeddings = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "embeddings"
)
os.makedirs(dir_embeddings, exist_ok=True)


########## Condition 1 ##########
# Embedding model: sentence-transformers/all-MiniLM-L6-v2
# Chunk: 每一條法條
# Overlap: 無
# Output file: laws_embedding_1chunk_0overlap_MiniLM.json
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


########## Condition 2 ##########
# Embedding model: lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct
# Chunk: 每一條法條
# Overlap: 無
# Output file: laws_embedding_1chunk_0overlap_Llama.json
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


########## Condition 3 ##########
# Embedding model: lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct
# Chunk: 每三條法條
# Overlap: 一條法條
# Output file: laws_embedding_3chunk_1overlap_Llama.json
fir_emb_llama_3c1o = os.path.join(
    dir_embeddings,
    "laws_embedding_3chunk_1overlap_Llama.json"
)

# Read the input data
with open(fil_laws_path, 'r', encoding='utf-8') as f:
    lines_con3 = [line.strip() for line in f if line.strip()]

# Process text lengths of input data
processed_lines_con3 = ProcessLawText(
    lines=lines_con3,
    max_length=401
)

# Generate chunks
chunks_con3 = GenerateChunksWithOverlap(
    lines=processed_lines_con3,
    chunk_size=3,
    overlap_size=1
)

# Generate embeddings
model_name = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
ChunkEmbeddingByLlama(
    chunks=chunks_con3,
    model_name=model_name,
    output_path=fir_emb_llama_3c1o
)


########## Condition 4 ##########
# Embedding model: lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct
# Chunk: 每1000個字
# Overlap: 每200個字 (20%)
# Output file: laws_embedding_1000chunk_200overlap_Llama.json
fir_emb_llama_1000c200o = os.path.join(
    dir_embeddings,
    "laws_embedding_1000chunk_200overlap_Llama.json"
)

# Read the input data
with open(fil_laws_path, 'r', encoding='utf-8') as f:
    lines_con4 = [line.strip() for line in f if line.strip()]

# Process text lengths of input data
processed_lines_con4 = ProcessLawText(
    lines=lines_con4,
    max_length=401
)

# Generate chunks
chunks_con4 = GenerateChunksWithOverlapChar(
    lines=processed_lines_con4,
    chunk_size=1000,
    overlap_ratio=0.2
)

# Generate embeddings
model_name = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
ChunkEmbeddingByLlama(
    chunks=chunks_con4,
    model_name=model_name,
    output_path=fir_emb_llama_1000c200o
)


########## Condition 5 ##########
# Embedding model: lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct
# Chunk: 每四條法條
# Overlap: 二條法條
# Output file: laws_embedding_4chunk_2overlap_Llama.json
fir_emb_llama_4chunk_2overlap = os.path.join(
    dir_embeddings,
    "laws_embedding_4chunk_2overlap_Llama.json"
)

# Read the input data
with open(fil_laws_path, 'r', encoding='utf-8') as f:
    lines_con5 = [line.strip() for line in f if line.strip()]

# Process text lengths of input data
processed_lines_con5 = ProcessLawText(
    lines=lines_con5,
    max_length=401
)

# Generate chunks
chunks_con5 = GenerateChunksWithOverlap(
    lines=processed_lines_con5,
    chunk_size=4,
    overlap_size=2
)

# Generate embeddings
model_name = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
ChunkEmbeddingByLlama(
    chunks=chunks_con5,
    model_name=model_name,
    output_path=fir_emb_llama_4chunk_2overlap
)
