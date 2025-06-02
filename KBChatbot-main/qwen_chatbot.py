# qwen_chatbot.py

import os
import pickle
import faiss
import torch
import numpy as np
import re
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# === 1. Chunking ===
def simple_chunk_text(text, chunk_size=500, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# === 2. Load PDFs and Chunk ===
folder_path = r"C:\Users\pushk\OneDrive\Desktop\KBChatbot-main\DocumentDatasetMultiple"
def load_and_chunk_pdfs(folder_path):
    chunks, sources = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "rb") as f:
                reader = PdfReader(f)
                raw_text = "".join(page.extract_text() or "" for page in reader.pages)
                split_texts = simple_chunk_text(raw_text)
                chunks.extend(split_texts)
                sources.extend([filename] * len(split_texts))
    return chunks, sources

chunks, sources = load_and_chunk_pdfs(folder_path)

# === 3. Embedding and FAISS Index ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, batch_size=8, convert_to_numpy=True).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === 4. Search Function ===
def search(query, top_k=5):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# === 5. Load Qwen Model ===
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
).eval()

# === 6. Qwen Chat Logic ===
def chat_with_qwen(user_input, history=[]):
    messages = []
    for role, content in history:
        if role == "user":
            messages.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    messages.append(f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant")

    prompt = "\n".join(messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cpu()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return output_text

def clean_qwen_response(raw_output):
    return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

# === 7. Streamlit UI ===
st.set_page_config(page_title="Qwen Chatbot", layout="centered")
st.title("🧠 Qwen Chatbot Interface")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    raw_response = chat_with_qwen(user_input, st.session_state.chat_history)
    cleaned_response = clean_qwen_response(raw_response)
    st.session_state.chat_history.append(("assistant", cleaned_response))

for sender, msg in st.session_state.chat_history:
    st.chat_message(sender).markdown(msg)


# qwen_chatbot.py (GPU-optimized version)

# import os
# import pickle
# import faiss
# import torch
# import numpy as np
# import re
# import streamlit as st
# from PyPDF2 import PdfReader
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer

# # === 1. Chunking ===
# def simple_chunk_text(text, chunk_size=500, chunk_overlap=100):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - chunk_overlap
#     return chunks

# # === 2. Load PDFs and Chunk ===
# folder_path = r"C:\Users\pushk\OneDrive\Desktop\KBChatbot-main\DocumentDatasetMultiple"
# def load_and_chunk_pdfs(folder_path):
#     chunks, sources = [], []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             full_path = os.path.join(folder_path, filename)
#             with open(full_path, "rb") as f:
#                 reader = PdfReader(f)
#                 raw_text = "".join(page.extract_text() or "" for page in reader.pages)
#                 split_texts = simple_chunk_text(raw_text)
#                 chunks.extend(split_texts)
#                 sources.extend([filename] * len(split_texts))
#     return chunks, sources

# chunks, sources = load_and_chunk_pdfs(folder_path)

# # === 3. Embedding and FAISS Index ===
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = embedder.encode(chunks, batch_size=8, convert_to_numpy=True).astype("float32")
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)

# # === 4. Search Function ===
# def search(query, top_k=5):
#     query_embedding = embedder.encode([query]).astype("float32")
#     distances, indices = index.search(query_embedding, top_k)
#     results = [chunks[i] for i in indices[0]]
#     return results

# # === 5. Load Qwen Model on GPU ===
# model_id = "Qwen/Qwen1.5-0.5B-Chat"

# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cuda",  # Force GPU usage
#     trust_remote_code=True,
#     torch_dtype=torch.float16  # Use half-precision for speed
# ).eval()

# # === 6. Qwen Chat Logic with RAG ===
# def chat_with_qwen(user_input, history=[]):
#     # Retrieve top relevant chunks from the documents
#     context_chunks = search(user_input, top_k=3)
#     context_text = "\n".join(context_chunks)

#     # Build system prompt with context
#     system_message = f"<|im_start|>system\nUse the following context to answer the user question as accurately as possible.\n\n{context_text}<|im_end|>"

#     # Build full message history with user and assistant
#     messages = [system_message]
#     for role, content in history:
#         if role == "user":
#             messages.append(f"<|im_start|>user\n{content}<|im_end>")
#         elif role == "assistant":
#             messages.append(f"<|im_start|>assistant\n{content}<|im_end>")
#     messages.append(f"<|im_start|>user\n{user_input}<|im_end>\n<|im_start|>assistant")

#     prompt = "\n".join(messages)
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids,
#             max_new_tokens=128,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.95,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
#     return output_text

# def clean_qwen_response(raw_output):
#     return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

# # === 7. Streamlit UI ===
# st.set_page_config(page_title="Qwen Chatbot", layout="centered")
# st.title("🧠 Qwen Chatbot Interface")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_input = st.chat_input("Ask me anything...")

# if user_input:
#     st.session_state.chat_history.append(("user", user_input))
#     with st.spinner("Thinking..."):
#         raw_response = chat_with_qwen(user_input, st.session_state.chat_history)
#         cleaned_response = clean_qwen_response(raw_response)
#     st.session_state.chat_history.append(("assistant", cleaned_response))

# for sender, msg in st.session_state.chat_history:
#     st.chat_message(sender).markdown(msg)
