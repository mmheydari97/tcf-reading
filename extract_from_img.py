import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import PIL.Image
import json
from streamlit_paste_button import paste_image_button as pbutton
import io
import os
from dotenv import load_dotenv
import uuid
import hashlib
import base64
from openai import OpenAI

# ==========================================
# 0. SETUP & CONFIG
# ==========================================
load_dotenv()  # Load environment variables from .env file

# --- Helper to prevent duplicates ---
def get_image_hash(image):
    """Generates a simple hash for image to prevent duplicates"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format if image.format else 'PNG')
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

# --- Helper to convert PIL Image to base64 ---
def image_to_base64(image):
    """Converts PIL Image to base64 string for OpenRouter API"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 1. PYDANTIC SCHEMA
# ==========================================
class ExamQuestion(BaseModel):
    id: int = Field(..., description="The sequential number of the question")
    readingText: Optional[str] = Field(None, description="The reading passage. Null if standalone.")
    text: str = Field(..., description="The actual question text")
    options: List[str] = Field(..., min_items=4, max_items=4, description="Exactly 4 options")
    correctIndex: int = Field(..., ge=0, le=3, description="Index of the correct answer (0-3)")
    explanation: str = Field(..., description="Explanation in English describing the logic")

class ExamOutput(BaseModel):
    examTitle: str
    timeLimitMinutes: int
    questions: List[ExamQuestion]

# ==========================================
# 2. LANGGRAPH LOGIC
# ==========================================
class GraphState(TypedDict):
    image: Any
    question_id: int
    api_key: str
    provider: str  # "google" or "openrouter"
    model: str
    raw_response: str
    structured_data: Optional[ExamQuestion]
    error: Optional[str]

def extract_node(state: GraphState):
    prompt_text = f"""
    You are an expert French Tutor. Analyze this TCF exam image (ID: {state['question_id']}).
    IGNORE human marks. SOLVE the question yourself.
    Extract and return ONLY valid JSON with these fields:
    - id: {state['question_id']}
    - readingText: the reading passage text or null if standalone
    - text: the actual question text
    - options: array of exactly 4 options
    - correctIndex: index of correct answer (0-3)
    - explanation: explanation in French describing the logic
    
    Return ONLY the JSON object, no markdown formatting.
    """
    
    try:
        if state["provider"] == "google":
            # --- Google Gemini API ---
            genai.configure(api_key=state["api_key"])
            model = genai.GenerativeModel(
                model_name=state["model"],
                generation_config={"response_mime_type": "application/json", "temperature": 0.0}
            )
            response = model.generate_content([prompt_text, state["image"]])
            return {"raw_response": response.text}
        
        elif state["provider"] == "openrouter":
            # --- OpenRouter API (OpenAI-compatible) ---
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=state["api_key"],
            )
            
            # Convert image to base64
            img_base64 = image_to_base64(state["image"])
            
            response = client.chat.completions.create(
                model=state["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,
            )
            
            raw_text = response.choices[0].message.content
            # Clean up potential markdown formatting
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return {"raw_response": raw_text.strip()}
        
        else:
            return {"error": f"Unknown provider: {state['provider']}"}
            
    except Exception as e:
        return {"error": str(e)}

def validate_node(state: GraphState):
    if state.get("error"):
        return state
    try:
        data_dict = json.loads(state["raw_response"])
        data_dict['id'] = state['question_id']
        validated_obj = ExamQuestion(**data_dict)
        return {"structured_data": validated_obj}
    except (json.JSONDecodeError, ValidationError) as e:
        return {"error": f"Validation Error: {str(e)}"}

workflow = StateGraph(GraphState)
workflow.add_node("extract", extract_node)
workflow.add_node("validate", validate_node)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", END)
app_graph = workflow.compile()

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="TCF Exam Digitizer", layout="wide")

# --- CSS for Aesthetics ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; }
    .img-card { border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; margin-bottom: 10px; background-color: #f9f9f9; }
    .remove-btn { color: red; font-weight: bold; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

st.title("📄 TCF Exam Screenshot Processor")

# --- SESSION STATE INITIALIZATION ---
if "gallery" not in st.session_state:
    st.session_state.gallery = []  # Stores dicts: {'id': uuid, 'img': PIL_Image, 'hash': str}
if "processed_questions" not in st.session_state:
    st.session_state.processed_questions = []
if "processed_paste_hashes" not in st.session_state:
    st.session_state.processed_paste_hashes = set()  # Track paste operations we've already handled

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # --- Provider Selection ---
    st.subheader("🔌 Service Provider")
    provider = st.radio(
        "Choose Provider",
        options=["google", "openrouter"],
        format_func=lambda x: "Google Gemini" if x == "google" else "OpenRouter",
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # --- Model Selection (default changes based on provider) ---
    default_models = {
        "google": "gemini-2.5-flash",
        "openrouter": "google/gemini-2.0-flash-exp:free"
    }
    model_name = st.text_input(
        "Model Name",
        value=default_models[provider],
        help="Enter the model identifier for the selected provider"
    )
    
    st.divider()
    
    # --- API Key Section ---
    st.subheader("🔑 API Key")
    
    # Get environment keys
    env_google_key = os.getenv("GOOGLE_API_KEY")
    env_openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # Determine which env key to use based on provider
    env_key = env_google_key if provider == "google" else env_openrouter_key
    env_var_name = "GOOGLE_API_KEY" if provider == "google" else "OPENROUTER_API_KEY"
    
    user_key = st.text_input(
        f"API Key ({provider.title()})",
        type="password",
        placeholder=f"Leave empty to use {env_var_name} from .env",
        label_visibility="collapsed"
    )
    
    # Use user key if provided, otherwise fall back to env key
    api_key = user_key if user_key else env_key
    
    if api_key:
        st.success(f"✅ API Key Loaded for {provider.title()}")
    else:
        st.error(f"❌ No API Key Found (set {env_var_name} in .env or enter above)")
    
    st.divider()
    
    # --- Exam Settings ---
    st.subheader("📝 Exam Settings")
    exam_title = st.text_input("Exam Title", "TCF Entraînement 1")
    time_limit = st.number_input("Time Limit (Minutes)", value=60)

# ==========================================
# INPUT PANEL (Unified)
# ==========================================
st.markdown("### 1. Add Images")
input_container = st.container(border=True)

with input_container:
    col_up, col_paste = st.columns([1, 1])
    
    # --- A. Upload Logic ---
    with col_up:
        uploaded_files = st.file_uploader(
            "📂 Upload Files",
            accept_multiple_files=True,
            type=['png', 'jpg', 'webp'],
            label_visibility="collapsed"
        )
        if uploaded_files:
            for f in uploaded_files:
                img = PIL.Image.open(f)
                img_hash = get_image_hash(img)
                # Check duplication
                if not any(d['hash'] == img_hash for d in st.session_state.gallery):
                    st.session_state.gallery.append({'id': str(uuid.uuid4()), 'img': img, 'hash': img_hash})

    # --- B. Paste Logic ---
    with col_paste:
        st.write("")  # Spacer to align with uploader
        paste_result = pbutton(
            label="📋 Paste from Clipboard",
            text_color="#ffffff",
            background_color="#4CAF50",
            hover_background_color="#45a049",
        )
        
        if paste_result.image_data is not None:
            img = paste_result.image_data
            img_hash = get_image_hash(img)
            # Only process if this paste hasn't been handled yet
            if img_hash not in st.session_state.processed_paste_hashes:
                st.session_state.processed_paste_hashes.add(img_hash)
                if not any(d['hash'] == img_hash for d in st.session_state.gallery):
                    st.session_state.gallery.append({'id': str(uuid.uuid4()), 'img': img, 'hash': img_hash})

# ==========================================
# GALLERY DISPLAY (With Remove Buttons)
# ==========================================
if st.session_state.gallery:
    st.markdown(f"### 🖼️ Selected Images ({len(st.session_state.gallery)})")
    
    # Display in a grid of 5 columns
    cols = st.columns(5)
    
    for idx, item in enumerate(st.session_state.gallery):
        col = cols[idx % 5]
        with col:
            st.image(item['img'], use_container_width=True)
            if st.button("❌ Remove", key=f"del_{item['id']}"):
                st.session_state.gallery.pop(idx)
                st.rerun()

# ==========================================
# PROCESSING LOGIC
# ==========================================
st.divider()

# Show current configuration
with st.expander("📊 Current Configuration", expanded=False):
    st.write(f"**Provider:** {provider.title()}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**API Key:** {'✅ Set' if api_key else '❌ Not Set'}")

if st.button("🚀 Start Processing", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please provide an API Key.")
    elif not st.session_state.gallery:
        st.warning("Please add at least one image.")
    else:
        st.session_state.processed_questions = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(st.session_state.gallery)
        
        for i, item in enumerate(st.session_state.gallery):
            status_text.text(f"Processing Image {i + 1}/{total} via {provider.title()}...")
            progress_bar.progress(i / total)
            
            inputs = {
                "image": item['img'],
                "question_id": i + 1,
                "api_key": api_key,
                "provider": provider,
                "model": model_name,
                "raw_response": "",
                "structured_data": None,
                "error": None
            }
            
            result = app_graph.invoke(inputs)
            
            if result.get("error"):
                st.error(f"Error on Image {i+1}: {result['error']}")
            elif result.get("structured_data"):
                st.session_state.processed_questions.append(result["structured_data"])
        
        progress_bar.progress(100)
        status_text.text("Processing Complete!")

# ==========================================
# RESULTS & DOWNLOAD
# ==========================================
if st.session_state.processed_questions:
    st.divider()
    st.subheader("🎉 Results")
    
    # Convert Pydantic objects to pure dicts for JSON serialization
    questions_dicts = [q.model_dump() for q in st.session_state.processed_questions]
    
    final_exam = ExamOutput(
        examTitle=exam_title,
        timeLimitMinutes=time_limit,
        questions=questions_dicts
    )
    
    json_str = final_exam.model_dump_json(indent=2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.json(json_str, expanded=False)
    with col2:
        st.download_button(
            "⬇️ Download exam.json",
            data=json_str,
            file_name="exam.json",
            mime="application/json",
            type="primary"
        )
