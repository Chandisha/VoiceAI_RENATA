import torch
import numpy as np
import faiss
# We need to install librosa for audio processing
import librosa
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    # New import for STT model
    AutoModelForSpeechSeq2Seq,
)
from sentence_transformers import SentenceTransformer

# --- 0. Configuration & File Paths ---
# Model IDs
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"

# STT Model: Using a smaller, faster English-focused model (Whisper Base) is often adequate.
STT_MODEL_ID = "openai/whisper-base"

# ENGLISH EMBEDDING MODEL: Switched back to the highly efficient, English-only model (384 dimensions)
# NOTE: YOU MUST RE-RUN THE INDEXING SCRIPT WITH THIS MODEL BEFORE RUNNING THIS INFERENCE FILE!
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" 

# RAG Index Files (Output from create_index.py)
INDEX_FILE_PATH = "wikitext_faiss_index.bin"
CORPUS_FILE_PATH = "wikitext_corpus_chunks.npy"

# **Placeholder for Audio File**
# IMPORTANT: You must replace this path with your actual audio file path (e.g., .wav, .mp3)
AUDIO_FILE_PATH = r"C:\Users\anilsagar\Desktop\SLM_rag_project\sample-000021.mp3" 

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    # Changed to warning, allowing CPU if absolutely necessary for STT/Embedder loading
    print("Warning: CUDA not available. RAG may run very slow on CPU.")
    # But Qwen is too slow, so we keep the RuntimeError for the LLM part
    pass
print(f"Running on device: {device}")


# --- 1. Load RAG Components (Index & Embedder) ---
print("--- 1. Loading RAG Components ---")

# Load the raw text chunks (corpus)
corpus_chunks = np.load(CORPUS_FILE_PATH, allow_pickle=True)
print(f"Loaded {len(corpus_chunks)} raw text chunks.")

# Load the FAISS Index
faiss_index = faiss.read_index(INDEX_FILE_PATH)
print(f"Loaded FAISS index (Dimension: {faiss_index.d}).")

# Load the Embedding Model (NOW ENGLISH-ONLY)
# Set to 'cpu' for the embedder to reserve maximum VRAM for the LLM
embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device='cpu')
print(f"Loaded English Sentence Transformer ({EMBEDDING_MODEL_ID}) for retrieval on CPU.")


# --- 2. Load LLM and STT Models ---

# A. LLM Loading (Qwen)
print(f"\n--- 2A. Loading LLM ({MODEL_ID} 4-bit) ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Enforce CUDA for the large LLM
if device == "cpu":
    raise RuntimeError("Qwen 1.8B is too large to run efficiently on CPU. CUDA required.")
    
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("LLM loaded in 4-bit for inference.")

# B. STT Loading (Whisper Base/Large)
print(f"\n--- 2B. Loading STT Model ({STT_MODEL_ID}) ---")

# We use the AutoModelForSpeechSeq2Seq for the model path for faster loading
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(STT_MODEL_ID).to(device)

stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model=STT_MODEL_ID,
    # NOTE: Whisper Base is less memory intensive than large-v3, but we keep it on GPU
    device=device
)
print(f"STT Model loaded on device: {stt_pipeline.device}")


# --- 3. The STT Function (Transcribes ONLY in English) ---
def transcribe_audio(audio_path: str) -> str:
    """
    Converts an audio file into an English text string using the STT pipeline (transcription only).
    """
    try:
        print(f"Transcribing audio file: {audio_path}...")
        
        # KEY CHANGE: Set task to "transcribe" and specify language "en"
        result = stt_pipeline(
            audio_path,
            generate_kwargs={"task": "transcribe", "language": "en"}
        ) 
        
        transcribed_text = result["text"]
        print(f"Transcription successful: '{transcribed_text}'")
        return transcribed_text.strip()
    except Exception as e:
        print(f"Error during STT transcription: {e}")
        return ""


# --- 4. The RAG Query Function (English Logic) ---
def rag_query(query: str, k: int = 5):
    # a. Encode the user query
    # KEY CHANGE: Removed the "query: " prefix as it is not required for MiniLM-L6-v2
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)

    # b. Search the FAISS index (Retrieval)
    D, I = faiss_index.search(query_embedding, k) # D=distances, I=indices
    
    # c. Get the relevant context chunks
    retrieved_contexts = [corpus_chunks[i] for i in I[0]]
    combined_context = "\n---\n".join(retrieved_contexts)

    # d. Construct the Qwen prompt (Augmentation)
    prompt_messages = [
        {"role": "system", "content": f"You are an accurate, helpful assistant. Answer the user's question based ONLY on the provided context.\n\nContext: {combined_context}"},
        {"role": "user", "content": query},
    ]

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

# --- 5. Run Audio-Based RAG Query ---
if __name__ == "__main__":
    
    # 5A. STT STEP: Convert audio to text query
    stt_query = transcribe_audio(AUDIO_FILE_PATH)
    
    if not stt_query:
        print("\n❌ Failed to get a query from audio. Exiting.")
        exit()

    # 5B. RAG STEP: Use the transcribed text as the input query
    print(f"\n--- 5. Running RAG Query with STT Result ---")
    print(f"Query (from STT): {stt_query}")
    
    final_answer, context_list = rag_query(stt_query)
    
    print("\n✅ Final Generated Answer:")
    print(final_answer)
    
    print("\n📚 Context Chunks Retrieved:")
    for i, context in enumerate(context_list):
        print(f"Chunk {i+1}: {context[:150]}...") # Print first 150 chars
import torch
import numpy as np
import faiss
# We need to install librosa for audio processing
import librosa
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    # New import for STT model
    AutoModelForSpeechSeq2Seq,
)
from sentence_transformers import SentenceTransformer

# --- 0. Configuration & File Paths ---
# Model IDs
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"

# STT Model: Using a smaller, faster English-focused model (Whisper Base) is often adequate.
STT_MODEL_ID = "openai/whisper-base"

# ENGLISH EMBEDDING MODEL: Switched back to the highly efficient, English-only model (384 dimensions)
# NOTE: YOU MUST RE-RUN THE INDEXING SCRIPT WITH THIS MODEL BEFORE RUNNING THIS INFERENCE FILE!
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" 

# RAG Index Files (Output from create_index.py)
INDEX_FILE_PATH = "wikitext_faiss_index.bin"
CORPUS_FILE_PATH = "wikitext_corpus_chunks.npy"

# **Placeholder for Audio File**
# IMPORTANT: You must replace this path with your actual audio file path (e.g., .wav, .mp3)
AUDIO_FILE_PATH = r"C:\Users\anilsagar\Desktop\SLM_rag_project\sample-000021.mp3" 

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    # Changed to warning, allowing CPU if absolutely necessary for STT/Embedder loading
    print("Warning: CUDA not available. RAG may run very slow on CPU.")
    # But Qwen is too slow, so we keep the RuntimeError for the LLM part
    pass
print(f"Running on device: {device}")


# --- 1. Load RAG Components (Index & Embedder) ---
print("--- 1. Loading RAG Components ---")

# Load the raw text chunks (corpus)
corpus_chunks = np.load(CORPUS_FILE_PATH, allow_pickle=True)
print(f"Loaded {len(corpus_chunks)} raw text chunks.")

# Load the FAISS Index
faiss_index = faiss.read_index(INDEX_FILE_PATH)
print(f"Loaded FAISS index (Dimension: {faiss_index.d}).")

# Load the Embedding Model (NOW ENGLISH-ONLY)
# Set to 'cpu' for the embedder to reserve maximum VRAM for the LLM
embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device='cpu')
print(f"Loaded English Sentence Transformer ({EMBEDDING_MODEL_ID}) for retrieval on CPU.")


# --- 2. Load LLM and STT Models ---

# A. LLM Loading (Qwen)
print(f"\n--- 2A. Loading LLM ({MODEL_ID} 4-bit) ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Enforce CUDA for the large LLM
if device == "cpu":
    raise RuntimeError("Qwen 1.8B is too large to run efficiently on CPU. CUDA required.")
    
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("LLM loaded in 4-bit for inference.")

# B. STT Loading (Whisper Base/Large)
print(f"\n--- 2B. Loading STT Model ({STT_MODEL_ID}) ---")

# We use the AutoModelForSpeechSeq2Seq for the model path for faster loading
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(STT_MODEL_ID).to(device)

stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model=STT_MODEL_ID,
    # NOTE: Whisper Base is less memory intensive than large-v3, but we keep it on GPU
    device=device
)
print(f"STT Model loaded on device: {stt_pipeline.device}")


# --- 3. The STT Function (Transcribes ONLY in English) ---
def transcribe_audio(audio_path: str) -> str:
    """
    Converts an audio file into an English text string using the STT pipeline (transcription only).
    """
    try:
        print(f"Transcribing audio file: {audio_path}...")
        
        # KEY CHANGE: Set task to "transcribe" and specify language "en"
        result = stt_pipeline(
            audio_path,
            generate_kwargs={"task": "transcribe", "language": "en"}
        ) 
        
        transcribed_text = result["text"]
        print(f"Transcription successful: '{transcribed_text}'")
        return transcribed_text.strip()
    except Exception as e:
        print(f"Error during STT transcription: {e}")
        return ""


# --- 4. The RAG Query Function (English Logic) ---
def rag_query(query: str, k: int = 5):
    # a. Encode the user query
    # KEY CHANGE: Removed the "query: " prefix as it is not required for MiniLM-L6-v2
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)

    # b. Search the FAISS index (Retrieval)
    D, I = faiss_index.search(query_embedding, k) # D=distances, I=indices
    
    # c. Get the relevant context chunks
    retrieved_contexts = [corpus_chunks[i] for i in I[0]]
    combined_context = "\n---\n".join(retrieved_contexts)

    # d. Construct the Qwen prompt (Augmentation)
    prompt_messages = [
        {"role": "system", "content": f"You are an accurate, helpful assistant. Answer the user's question based ONLY on the provided context.\n\nContext: {combined_context}"},
        {"role": "user", "content": query},
    ]

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

# --- 5. Run Audio-Based RAG Query ---
if __name__ == "__main__":
    
    # 5A. STT STEP: Convert audio to text query
    stt_query = transcribe_audio(AUDIO_FILE_PATH)
    
    if not stt_query:
        print("\n❌ Failed to get a query from audio. Exiting.")
        exit()

    # 5B. RAG STEP: Use the transcribed text as the input query
    print(f"\n--- 5. Running RAG Query with STT Result ---")
    print(f"Query (from STT): {stt_query}")
    
    final_answer, context_list = rag_query(stt_query)
    
    print("\n✅ Final Generated Answer:")
    print(final_answer)
    
    print("\n📚 Context Chunks Retrieved:")
    for i, context in enumerate(context_list):
        print(f"Chunk {i+1}: {context[:150]}...") # Print first 150 chars
