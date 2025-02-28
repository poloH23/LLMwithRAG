import os
import json
import torch
from tqdm import tqdm
from typing import Optional
from lib.Utils import GetRoot
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


def GetHfToken() -> Optional[str]:
    # Load environment variables
    GetRoot()

    # Get ".token" file
    fil_token = os.environ.get("PROJECT_ROOT") + os.getenv("HF_TOKEN")

    # Get HuggingFace token
    if os.path.exists(fil_token):
        with open(fil_token, "r") as fil_token:
            hf_token = fil_token.read().strip()
            command = f"export HUGGINGFACE_TOKEN={hf_token}"
            os.system(command)
            return ">>> HuggingFace token applied."
    return None

def EmbeddingByMiniLM(
input_path: str,
        output_path: str,
        model_name: str,
        batch_size: int
    )-> None:
    """
        :param input_path: Path to input file (Laws file).
        :param output_path: Path to output file (Embedding result).
        :param model_name: Embedding model name (use MiniLM as default).
        :param batch_size: Batch size for SentenceTransformer.
        :return: None
        """
    # 加載模型，默認使用"sentence-transformers/all-MiniLM-L6-v2"
    if model_name is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # 讀取法條文本
    with open(input_path, "r", encoding="utf-8") as f:
        lis_line = [line.strip() for line in f.readlines() if line.strip()]

    # 批次產生詞向量
    embeddings = []
    str_desc = "Generating embeddings"
    for i in tqdm(range(0, len(lis_line), batch_size), desc=str_desc):
        batch = lis_line[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True).cpu().numpy()
        embeddings.extend([{"text": text, "embedding": emb.tolist()} for text, emb in zip(batch, batch_embeddings)])

    # 保存為JSON格式
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)

    print(f">>> The embedding file has been saved: {output_path}")
    return None


def EmbeddingByLlama(
        input_path: str,
        output_path: str,

        model_name: str,
        batch_size: int
) -> None:
    """
    :param input_path: Path to input file (Laws file).
    :param output_path: Path to output file (Embedding result).
    :param model_name: Embedding model name (use MiniLM as default).
    :param batch_size: Batch size for SentenceTransformer.
    :return: None
    """
    if torch.cuda.is_available():
        print("CUDA 可用，正在使用 GPU：", torch.cuda.get_device_name(0))
    else:
        print("CUDA 不可用，正在使用 CPU")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(
        "cuda" if torch.cuda.is_available() else "cpu")

    # 讀取法條文本
    with open(input_path, "r", encoding="utf-8") as f:
        lis_line = [line.strip() for line in f.readlines() if line.strip()]

    # 批次產生詞向量
    embeddings = []
    for i in tqdm(range(0, len(lis_line), batch_size), desc="Generating embeddings"):
        batch = lis_line[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model(**tokens)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend([{"text": text, "embedding": emb.tolist()} for text, emb in zip(batch, batch_embeddings)])

    # 保存為JSON格式
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)

    print(f">>> The embedding file has been saved: {output_path}")
