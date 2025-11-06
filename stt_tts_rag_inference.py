import torch
import os
import time
import uuid

# Hugging Face Transformers and Sentence Transformers for RAG components
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# FAISS for vector search
from faiss import read_index, Index

# gTTS for Text-to-Speech (requires pip install gTTS)
from gtts import gTTS 

# librosa for audio loading (requires FFmpeg to be on the system PATH)
import numpy as np # Added numpy import for loading .npy corpus file
import librosa 

# --- Configuration (UPDATE THESE PATHS AS NEEDED) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NOTE: Using the file that was successfully transcribed in the previous step
AUDIO_FILE_PATH = "C:\\Users\\anilsagar\\Desktop\\SLM_rag_project\\sample-000021.mp3" 
TTS_OUTPUT_FILE = "rag_response.mp3"
# --- CORRECTED PATHS to match your directory structure ---
FAISS_INDEX_PATH = "wikitext_faiss_index.bin"
CORPUS_FILE_PATH = "wikitext_corpus_chunks.npy"

# Model Configurations
LLM_MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
STT_MODEL_NAME = "openai/whisper-base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- 1. RAG Component Initialization ---
def load_rag_components():
    """Loads all models, tokenizer, index, and raw data."""
    print(f"Running on device: {DEVICE}")
    print("--- 1. Loading RAG Components ---")

    # Load raw text chunks (using the .npy file created by create_index.py)
    try:
        raw_chunks_np = np.load(CORPUS_FILE_PATH, allow_pickle=True)
        raw_chunks = raw_chunks_np.tolist()
        print(f"Loaded {len(raw_chunks)} raw text chunks from {CORPUS_FILE_PATH}.")
    except Exception as e:
        print(f"Error loading raw text chunks: {e}")
        return None, None, None, None, None, None, None
        
    # Load FAISS Index (using the correct filename)
    try:
        index = read_index(FAISS_INDEX_PATH)
        print(f"Loaded FAISS index (Dimension: {index.d}).")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None, None, None, None, None, None, None

    # Load Sentence Transformer (Embedding Model)
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # Embedder is typically run on CPU to save VRAM for the LLM
        embedder.to("cpu")
        print(f"Loaded English Sentence Transformer ({EMBEDDING_MODEL_NAME}) for retrieval on CPU.")
    except Exception as e:
        print(f"Error loading Sentence Transformer: {e}")
        return index, None, None, None, None, None, None

    # Load LLM (Quantized for GPU efficiency)
    print("\n--- 2A. Loading LLM (Qwen/Qwen1.5-1.8B-Chat 4-bit) ---")
    try:
        if DEVICE == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config
            )
            actual_device = model.device
            print(f"Device set to use {actual_device}")
        else:
            model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
            print("Device set to use cpu")

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        print("LLM loaded in 4-bit for inference.")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return index, embedder, None, None, None, None, None

    # Load STT Model (Whisper)
    print("\n--- 2B. Loading STT Model (openai/whisper-base) ---")
    try:
        stt_model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_NAME)
        stt_processor = WhisperProcessor.from_pretrained(STT_MODEL_NAME)
        stt_model.to(DEVICE)
        print(f"STT Model loaded on device: {DEVICE}")
    except Exception as e:
        print(f"Error loading STT Model: {e}")
        return index, embedder, model, tokenizer, None, None, None

    return index, embedder, model, tokenizer, stt_model, stt_processor, raw_chunks

# --- 2. STT Function (Audio to Text) ---
def transcribe_audio(audio_path, stt_model, stt_processor):
    """Transcribes the given audio file using the Whisper model."""
    print(f"Transcribing audio file: {audio_path}...")
    try:
        # librosa handles audio loading and resampling (requires FFmpeg)
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process the audio data
        input_features = stt_processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        input_features = input_features.to(stt_model.device)

        # Generate the transcription
        predicted_ids = stt_model.generate(
            input_features, 
            forced_decoder_ids=stt_processor.get_decoder_prompt_ids(language="english", task="transcribe")
        )

        # Decode the IDs to text
        transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"Transcription successful: '{transcription}'")
        return transcription

    except FileNotFoundError:
        print(f"Error during STT transcription: [Errno 2] No such file or directory: '{audio_path}'")
        return None
    except Exception as e:
        print(f"Error during STT transcription: {e}")
        return None

# --- 3. RAG Query Function (Text Generation) ---
def rag_query(query, index, embedder, llm_model, llm_tokenizer, raw_chunks, top_k=5):
    """
    Performs the RAG query: Embed query, search index, retrieve context, generate answer.
    """
    # Embed the query
    query_vector = embedder.encode(query)
    query_vector = query_vector.astype('float32').reshape(1, -1)

    # Search the FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Retrieve the context text
    context_chunks = [raw_chunks[i] for i in indices[0]]
    context_text = "\n\n".join(context_chunks)

    # Prepare the Prompt for Qwen
    system_prompt = (
        "You are an intelligent RAG chatbot. Your task is to answer the user's "
        "query based ONLY on the provided context below. Do not use external knowledge. "
        "If the context does not contain the answer, state that you could not find "
        "enough information in the provided context."
    )

    # Use Qwen's chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n---\n{context_text}\n---\n\nUser Query: {query}"}
    ]

    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate the Answer
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
    )
    
    # Remove the prompt tokens from the generated sequence
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]

    response = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return response.strip(), context_chunks

# --- 4. TTS Function (Text to Audio) ---
def text_to_audio(text: str, output_filename: str):
    """
    Converts a text string to an MP3 audio file using gTTS.
    """
    try:
        print(f"\n🎧 Generating TTS audio response to: {output_filename}...")
        
        # Clean up the text (replaces @-@ with a hyphen for better pronunciation)
        cleaned_text = text.replace("@-@", "-") 
        
        # Create a gTTS object
        tts = gTTS(text=cleaned_text, lang='en', tld='com', slow=False)
        
        # Save the audio file
        tts.save(output_filename)
        
        print(f"✅ TTS Audio saved successfully to {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"❌ Error during TTS generation: {e}")


# --- 5. Main Execution ---
if __name__ == "__main__":
    
    # Load all components (RAG, LLM, STT)
    # This call initializes the models and returns the handles (index, embedder, etc.)
    index, embedder, llm_model, llm_tokenizer, stt_model, stt_processor, raw_chunks = load_rag_components()

    # Check if all core components loaded successfully
    if not all([index, embedder, llm_model, llm_tokenizer, stt_model, stt_processor, raw_chunks]):
        print("\n--- ERROR: Failed to load all components. Exiting. ---")
        exit()
    
    # STAGE 1: STT (Audio to Text)
    # NOTE: These functions must now receive the loaded models/processors as arguments
    stt_query = transcribe_audio(AUDIO_FILE_PATH, stt_model, stt_processor)
    
    if not stt_query:
        print("\n❌ Failed to get a query from audio. Exiting.")
        exit()

    # STAGE 2: RAG (Text to Answer)
    print(f"\n--- 6. Running RAG Query with STT Result ---")
    print(f"Query (from STT): {stt_query}")
    
    final_answer, context_list = rag_query(stt_query, index, embedder, llm_model, llm_tokenizer, raw_chunks)
    
    print("\n✅ Final Generated Answer:")
    print(final_answer)
    
    print("\n📚 Context Chunks Retrieved:")
    for i, context in enumerate(context_list):
        display_context = context.replace("@-@", "-") 
        print(f"Chunk {i+1}: {display_context[:150]}...")
        
    # STAGE 3: TTS (Answer to Audio)
    text_to_audio(final_answer, TTS_OUTPUT_FILE)
