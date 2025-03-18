# RAG-driven Taiwan Law Chatbot

## 📌 Project Overview
The **RAG-driven Taiwan Law Chatbot** is a legal question-answering chatbot designed to retrieve and generate legal information from Taiwan’s regulatory texts. It integrates **Retrieval-Augmented Generation (RAG)** with **Large Language Models (LLM)** to enhance legal knowledge retrieval and provide accurate legal responses.

## ✨ Features
- **Legal Text Web Crawling**: Automatically extracts popular legal texts from Taiwan’s Laws & Regulations Database.
- **Embedding Generation**: Converts legal text into searchable vector embeddings using **Taiwan Legal Fine-tuned Llama** and **MiniLM**.
- **RAG-powered Question Answering**: Uses different **LLM models (Llama-3.2, Taiwan Legal Fine-tuned Llama-3.2)** to generate responses.
- **LINE Chatbot Integration**: Deploys the chatbot as a **LINE Bot** using `Flask` and `ngrok` for real-time user interactions.

## 🛠 Installation
### Prerequisites
- Python 3.9+
- pip
- Virtual Environment (optional but recommended)

### Install Environment
* Create a Conda environment using default name.
```bash
git clone https://github.com/your-github-username/RAG-Taiwan-Law-Chatbot.git
cd RAG-Taiwan-Law-Chatbot
conda env create -f ./bak/environment.yml
```

* Create a Conda environment using default name.
```bash
git clone https://github.com/your-github-username/RAG-Taiwan-Law-Chatbot.git
cd RAG-Taiwan-Law-Chatbot
conda env create -n my_env -f ./bak/environment.yml
```

### Install Dependencies
```bash
git clone https://github.com/your-github-username/RAG-Taiwan-Law-Chatbot.git
cd RAG-Taiwan-Law-Chatbot
pip install -r requirements.txt
```

## 🚀 Usage
### 1、Web Crawling for Legal Texts
```bash
python -m main.laws_web_crawling
```

### 2、Generate Embeddings
```bash
python -m main.laws_embedding
```

### 3、Generate Embeddings for Comparing
* Embedding strategy comparison.
```bash
python -m main.RAG_legal_chatbot_cmp
```

### 4、Run the RAG Chatbot Locally
* Compare the effects of RAG
```bash
python -m main.RAG_legal_chatbot
```

### 5、Deploy LINE Chatbot
* Run the command below and copy the 'Ngrok public URL' on the page of LINE Developers.
```bash
python -m main.legal_linebot_local_ver
```

## 📊 Model Details
| Model | Description |
|------|------------|
| `meta-llama/Llama-3.2-3B-Instruct` | General-purpose LLM optimized for QA tasks. |
| `lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct` | Fine-tuned model for Taiwan legal texts. |

## 📂 Dataset
- **Legal Corpus**: Extracted from Taiwan's National Laws Database.
- **Embedding Storage**: Available in `data/embeddings/` with multiple chunking and overlap configurations.

## 🤝 Contribution
Feel free to contribute via:
- **Bug Reports**: Open an issue.
- **Feature Requests**: Submit a pull request.
- **Dataset Updates**: Help enhance legal corpus coverage.

## 📜 License
This project is licensed under the GPL-3.0 License.
