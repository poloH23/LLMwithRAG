import json
import torch
import faiss
import numpy as np
from typing import Any
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel


def LoadEmbeddings(embedding_path: str) -> tuple:
    with open(embedding_path, 'r') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    embeddings = np.array([item["embedding"] for item in data], dtype="float32")
    return texts, embeddings

def CreateFaissIdx(embeddings: np.array) -> Any:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def SearchFaissIdx(
        model_name: str,
        query: str,
        idx: Any,
        texts: str,
        top_k: int,
        max_length: int
) -> list:
    # 載入詞向量產生模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    tokens = tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        query_embedding = model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    query_embedding = query_embedding.astype("float32")

    # 查詢相近的詞向量資料
    distances, indices = idx.search(query_embedding, top_k)
    results = [{"text": texts[idx], "score": distances[0][i]} for i, idx in enumerate(indices[0])]
    return results

def LlamaGenerateResponse(model_name: str, context: str, query: str, max_token: int) -> str:
    # 可使用的模型名稱
    lis_models = [
        "lianghsun/Llama-3.2-Taiwan-Legal-1B-Instruct",
        "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]

    # 確認所使用的模型名稱
    if model_name in lis_models:
        model_name = model_name
    else:
        model_name = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"

    # 檢查是否可以使用 GPU
    if torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = "auto"

    # 調用 LLM
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        temperature=0.5
    )

    # 提示詞
    prompt = f"""
    你是一個台灣法律諮詢顧問。請仔細閱讀使用者提供的事實案例，檢索台灣法律條文，並根據案例中所述行為識別可能涉及的法規與條文。請在回答中包含：
    1.引用相關法律名稱與條號。
    2.簡短解釋該法條的法律構成要件，對照案例事實與構成要件，簡短指出是否符合。
    3.最後簡短提供法律程序上的初步建議，如是否需律師協助或可行的解決途徑。
    4.如果無法確定或需要更多資訊請直接說明，不要提供未經查證的法律見解並在末尾提醒此為非正式法律意見。
    下列為檢索法條後的相關資訊: {context}
    """

    # 送入模型的段落，包含提示詞和使用者提問
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]

    # 模型推論生成的結果
    outputs = pipe(
        messages,
        max_new_tokens=max_token
    )

    # 取得模型結果並返回
    str_result = outputs[0]["generated_text"][-1]["content"]
    return str_result

def LlmWithoutRAG(
        llm_model: str,
        query: str,
        max_token: int
) -> str:
    # 取得回覆
    answer = LlamaGenerateResponse(model_name=llm_model, context='', query=query, max_token=max_token)
    return answer

def LlmWithRAG(
        embedding_model: str,
        embedding_dataset: str,
        llm_model: str,
        max_token: int,
        query: str,
        top_k: int
) -> str:
    # 加載向量數據
    texts, embeddings = LoadEmbeddings(embedding_path=embedding_dataset)

    # 建立 FAISS 索引
    index = CreateFaissIdx(embeddings=embeddings)

    # 檢索相關內容
    top_results = SearchFaissIdx(
        model_name=embedding_model,
        query=query,
        max_length=512,
        idx=index,
        texts=texts,
        top_k=top_k
    )

    # 將檢索內容合併為上下文
    context = "\n".join([result["text"] for result in top_results])

    # 取得回覆
    answer = LlamaGenerateResponse(model_name=llm_model, context=context, query=query, max_token=max_token)
    return answer
