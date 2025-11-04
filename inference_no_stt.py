import torch
import numpy as np
import faiss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from sentence_transformers import SentenceTransformer

# --- 0. Configuration & File Paths ---
# Model
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# RAG Index Files (Output from create_index.py)
INDEX_FILE_PATH = "wikitext_faiss_index.bin"
CORPUS_FILE_PATH = "wikitext_corpus_chunks.npy"
# GPU Check
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Please ensure your GPU setup is correct.")

# --- 1. Load RAG Components (Index & Embedder) ---
print("--- 1. Loading RAG Components ---")

# Load the raw text chunks (corpus)
corpus_chunks = np.load(CORPUS_FILE_PATH, allow_pickle=True)
print(f"Loaded {len(corpus_chunks)} raw text chunks.")

# Load the FAISS Index
faiss_index = faiss.read_index(INDEX_FILE_PATH)
print(f"Loaded FAISS index (Dimension: {faiss_index.d}).")

# Load the Embedding Model (Must be the same one used for indexing)
# Set to 'cpu' for the embedder to reserve maximum VRAM for the LLM
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Loaded Sentence Transformer for retrieval.")

# --- 2. Load Qwen 1.5-1.8B for Inference (4-bit) ---
# **FIXED PRINT STATEMENT** to accurately reflect the loaded model
print(f"\n--- 2. Loading LLM ({MODEL_ID} 4-bit) ---")

# 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Model and Tokenizer (This is pure inference loading)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Qwen models often use a specific chat template that doesn't strictly require these
# but setting them here for compatibility and best practice:
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("LLM loaded in 4-bit for inference.")

# --- 3. The RAG Query Function ---
def rag_query(query: str, k: int = 5):
    # a. Encode the user query
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    # FAISS requires a specific shape and dtype
    query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)

    # b. Search the FAISS index (Retrieval)
    D, I = faiss_index.search(query_embedding, k) # D=distances, I=indices
    
    # c. Get the relevant context chunks
    retrieved_contexts = [corpus_chunks[i] for i in I[0]]
    combined_context = "\n---\n".join(retrieved_contexts)

    # d. Construct the Qwen prompt (Augmentation)
    # Use the chat template format for Qwen models
    prompt_messages = [
        {"role": "system", "content": f"You are an accurate, helpful assistant. Answer the user's question based ONLY on the provided context.\n\nContext: {combined_context}"},
        {"role": "user", "content": query},
    ]

    # Apply the template to get the final input string
    prompt = generator.tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True 
    )

    # e. Generate the response
    outputs = generator(
        prompt,
        max_new_tokens=256,
        do_sample=False, 
        temperature=0.01, 
    )

    # f. Clean and return the result
    full_output = outputs[0]["generated_text"]
    response = full_output[len(prompt):].strip()
    
    return response, retrieved_contexts

# --- 4. Run Example Query ---
if __name__ == "__main__":
    test_query = "What is the primary product of the petrochemical industry and what is it used for?"

    print(f"\n--- 4. Running RAG Query ---")
    print(f"Query: {test_query}")
    
    final_answer, context_list = rag_query(test_query)
    
    print("\n✅ Final Generated Answer:")
    print(final_answer)
    
    print("\n📚 Context Chunks Retrieved:")
    for i, context in enumerate(context_list):
        print(f"Chunk {i+1}: {context[:150]}...") # Print first 150 chars
