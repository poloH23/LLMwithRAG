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
    return None


def SplitLongLawText(text: str, max_length=512) -> list:
    lis_result = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    return lis_result

def ProcessLawText(lines: list, max_length=512) -> list:
    processed_lines = []
    for line in lines:
        if len(line) > max_length:
            processed_lines.extend(SplitLongLawText(line, max_length))
        else:
            processed_lines.append(line)
    return processed_lines

def GenerateChunksWithOverlap(
        lines: list,
        chunk_size: int,
        overlap_size: int
) -> list:
    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk = " ".join(lines[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap_size
    return chunks


def GenerateChunksWithOverlapChar(
        lines: list,
        chunk_size: int,
        overlap_ratio: float
) -> list:
    chunks = []
    current_chunk = ''

    for line in lines:
        if len(current_chunk) + len(line) > chunk_size:
            chunks.append(current_chunk)
            overlap_chars = int(len(current_chunk) * overlap_ratio)
            current_chunk = current_chunk[-overlap_chars:] + " " + line
        else:
            current_chunk += " " + line if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def SaveEmbeddingsToJson(embeddings, output_path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)
    print(f"Embeddings saved to {output_path}")
    return None

def ChunkEmbeddingByLlama(
        chunks: list,
        model_name: str,
        output_path: str,
        batch_size=32,
        max_length=512
) -> None:
    if torch.cuda.is_available():
        print("CUDA 可用，正在使用 GPU：", torch.cuda.get_device_name(0))
    else:
        print("CUDA 不可用，正在使用 CPU")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(
        "cuda" if torch.cuda.is_available() else "cpu")

    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model(**tokens)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend([{"text": text, "embedding": emb.tolist()} for text, emb in zip(batch, batch_embeddings)])

    # Save embedding results
    if not embeddings:
        print(">>> No embeddings found.")
        return None
    else:
        SaveEmbeddingsToJson(
            embeddings=embeddings,
            output_path=output_path
        )
        return None


