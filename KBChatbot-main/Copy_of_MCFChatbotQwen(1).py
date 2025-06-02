#!/usr/bin/env python
# coding: utf-8

# In[9]:


pip install huggingface_hub[hf_xet]


# In[1]:


get_ipython().system('pip install PyPDF2')


# In[2]:


get_ipython().system('pip install transformers accelerate bitsandbytes sentence-transformers faiss-cpu')


# In[3]:


import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Define filenames
INDEX_FILE = "faiss_index.idx"
SOURCE_FILE = "sources.pkl"

if os.path.exists(INDEX_FILE) and os.path.exists(SOURCE_FILE):
    print("Loading existing FAISS index and sources...")
    index = faiss.read_index(INDEX_FILE)
    with open(SOURCE_FILE, "rb") as f:
        sources = pickle.load(f)
else:
    print("Index not found. Creating new FAISS index from PDFs...")


# In[4]:


def simple_chunk_text(text, chunk_size=500, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


folder_path = r"C:\Users\pushk\Desktop\chatbot\DocumentDatasetMultiple"

def load_and_chunk_pdfs(folder_path):
    import os
    from PyPDF2 import PdfReader

    chunks = []
    sources = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "rb") as f:
                reader = PdfReader(f)
                raw_text = ""
                for page in reader.pages:
                    raw_text += page.extract_text() or ""
                split_texts = simple_chunk_text(raw_text)
                chunks.extend(split_texts)
                sources.extend([filename] * len(split_texts))
    
    return chunks, sources

chunks, sources = load_and_chunk_pdfs(folder_path)


# In[5]:


from sentence_transformers import SentenceTransformer

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunks to embeddings
embeddings = embedder.encode(chunks, batch_size=8, convert_to_numpy=True)


# In[6]:


import faiss
import numpy as np

# Convert to float32 (required by FAISS)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


# In[7]:


def search(query, top_k=5):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results


# In[8]:


from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True).eval()


# In[9]:


from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)


# In[22]:


# def generate_response(prompt, max_tokens=256):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)



# def chat_with_qwen(messages, max_tokens=512):
#     # messages: list of dicts like [{"role": "user", "content": "Hi"}]
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# import re

# def clean_qwen_response(raw_output):
#     # Strip <think> tags and anything inside them
#     cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
#     return cleaned.strip()

def chat_with_qwen(user_input, history=[]):
    # Build conversation history into Qwen chat format
    messages = []
    for role, content in history:
        if role == "user":
            messages.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    # Add the new user message
    messages.append(f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant")

    # Join full prompt
    prompt = "\n".join(messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the new text generated
    output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return output_text


# In[14]:


# user_input = "What are the prerequisites of the subject Cyber Security?"
# response = generate_response(user_input)
# print(response)


chat_history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the prerequisites of the subject Cyber Security?"}
]


# In[15]:


response_raw = chat_with_qwen(chat_history)
response = clean_qwen_response(response_raw)
print(response)


# def chat_with_qwen(user_input, history=[]):
#     # Add your code here to call Qwen with streaming/history if needed
#     raw_response = your_qwen_generate_function(user_input, history)
#     return raw_response


# In[17]:


get_ipython().system('pip install streamlit')


# In[23]:


# import streamlit as st
# import re

# # Import your Qwen chatbot function
# # Replace this with your actual chatbot call
# def chat_with_qwen(user_input, history=[]):
#     # Dummy response for illustration; use your real Qwen API/chat call
#     raw_response = f"<think>Thinking...</think>\nThe answer to your question is:\n1. Point A\n2. Point B"
#     return raw_response

# # Clean Qwen output
# def clean_qwen_response(raw_output):
#     return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

# # Streamlit app
# st.set_page_config(page_title="Qwen Chatbot", layout="centered")
# st.title("🧠 Qwen Chatbot Interface")

# # Store chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # User input
# user_input = st.chat_input("Ask me anything...")

# if user_input:
#     st.session_state.chat_history.append(("user", user_input))

#     # Call the Qwen chatbot
#     raw_response = chat_with_qwen(user_input, st.session_state.chat_history)
#     cleaned_response = clean_qwen_response(raw_response)

#     st.session_state.chat_history.append(("assistant", cleaned_response))

# # Display chat history
# for sender, msg in st.session_state.chat_history:
#     if sender == "user":
#         st.chat_message("user").markdown(msg)
#     else:
#         st.chat_message("assistant").markdown(msg)


import streamlit as st
import re

def clean_qwen_response(raw_output):
    return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

# Streamlit app UI
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
    if sender == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)


# In[21]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




