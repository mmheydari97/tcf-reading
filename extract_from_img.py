import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import PIL.Image
import json
from streamlit_paste_button import paste_image_button as pbutton
import io

# ==========================================
# 1. PYDANTIC SCHEMA (The Guardrails)
# ==========================================
class ExamQuestion(BaseModel):
    id: int = Field(..., description="The sequential number of the question")
    readingText: Optional[str] = Field(
        None, 
        description="The reading passage or context. Null if standalone question."
    )
    text: str = Field(..., description="The actual question text")
    options: List[str] = Field(..., min_items=4, max_items=4, description="Exactly 4 options")
    correctIndex: int = Field(..., ge=0, le=3, description="Index of the correct answer (0-3)")
    explanation: str = Field(..., description="Explanation in French describing the logic")

class ExamOutput(BaseModel):
    examTitle: str
    timeLimitMinutes: int
    questions: List[ExamQuestion]

# ==========================================
# 2. LANGGRAPH STATE & NODES (The Engine)
# ==========================================

# Define the state that passes between nodes
class GraphState(TypedDict):
    image: Any           # PIL Image object
    question_id: int     # Current ID to assign
    api_key: str         # Google API Key
    raw_response: str    # Raw text from Gemini
    structured_data: Optional[ExamQuestion] # validated object
    error: Optional[str] # Error tracking

# NODE 1: The Extractor (Gemini)
def extract_node(state: GraphState):
    try:
        genai.configure(api_key=state["api_key"])
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={"response_mime_type": "application/json", "temperature": 0.0}
        )
        
        prompt = f"""
        You are an expert French Tutor. Analyze this TCF exam image (ID: {state['question_id']}).
        
        CRITICAL RULES:
        1. IGNORE any human handwriting, circles, or checkmarks.
        2. Read the text and SOLVE the question yourself to find the 'correctIndex'.
        3. Extract the 'readingText' if present (email, article, paragraph). If none, use null.
        4. Return a valid JSON object matching this schema:
           id, readingText, text, options (array of 4 strings), correctIndex (0-3), explanation.
        """
        
        response = model.generate_content([prompt, state["image"]])
        return {"raw_response": response.text}
    except Exception as e:
        return {"error": str(e)}

# NODE 2: The Validator (Pydantic)
def validate_node(state: GraphState):
    if state.get("error"):
        return state # Skip if previous step failed
    
    try:
        # 1. Parse string to dict
        data_dict = json.loads(state["raw_response"])
        
        # 2. Enforce ID consistency
        data_dict['id'] = state['question_id']
        
        # 3. Pydantic Validation
        validated_obj = ExamQuestion(**data_dict)
        
        return {"structured_data": validated_obj}
    except json.JSONDecodeError:
        return {"error": "AI failed to return valid JSON."}
    except ValidationError as e:
        return {"error": f"Schema Validation Failed: {e}"}

# BUILD THE GRAPH
workflow = StateGraph(GraphState)
workflow.add_node("extract", extract_node)
workflow.add_node("validate", validate_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", END)

app_graph = workflow.compile()

# ==========================================
# 3. STREAMLIT UI (The Dashboard)
# ==========================================
st.set_page_config(page_title="TCF Exam Digitizer", layout="wide")

st.title("📄 TCF Exam Screenshot Processor")
st.markdown("""
This tool uses **LangGraph** to orchestrate an AI workflow and **Pydantic** to guarantee the data structure.
""")

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google API Key", type="password")
    st.info("Get your key from [Google AI Studio](https://aistudio.google.com/)")
    
    exam_title = st.text_input("Exam Title", "TCF Entraînement 1")
    time_limit = st.number_input("Time Limit (Minutes)", value=60)

# --- Main Area: Dual Input Strategy ---
st.write("### 1. Upload or Paste")

# Container to hold the images to be processed
images_to_process = []

col1, col2 = st.columns(2)

with col1:
    st.info("Option A: Upload Files")
    uploaded_files = st.file_uploader(
        "Drop files here", 
        accept_multiple_files=True, 
        type=['png', 'jpg', 'jpeg', 'webp']
    )
    if uploaded_files:
        for f in uploaded_files:
            # Convert uploaded file to PIL Image immediately
            img = PIL.Image.open(f)
            images_to_process.append(img)

with col2:
    st.info("Option B: Paste from Clipboard")
    # This button grabs the image from your clipboard
    paste_result = pbutton(
        label="📋 Paste Image",
        text_color="#ffffff",
        background_color="#FF4B4B",
        hover_background_color="#FF0000",
    )
    
    if paste_result.image_data is not None:
        st.success("Image pasted successfully!")
        # The component returns a PIL image directly
        images_to_process.append(paste_result.image_data)
        # Show a preview of what was pasted
        st.image(paste_result.image_data, caption="Pasted Image", width=200)

if "processed_questions" not in st.session_state:
    st.session_state.processed_questions = []

# --- Processing Logic ---
if st.button("Start Processing", type="primary"):
    if not api_key:
        st.error("Please provide an API Key in the sidebar.")
    elif not images_to_process:
        st.warning("No images found! Please upload files or paste an image.")
    else:
        # Reset session state for new run
        st.session_state.processed_questions = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_images = len(images_to_process)
        
        for index, image in enumerate(images_to_process):
            status_text.text(f"Processing Image {index + 1}/{total_images}...")
            progress_bar.progress((index) / total_images)
            
            # --- INVOKE LANGGRAPH ---
            inputs = {
                "image": image,
                "question_id": index + 1,
                "api_key": api_key,
                "raw_response": "",
                "structured_data": None,
                "error": None
            }
            
            # Run the graph
            result_state = app_graph.invoke(inputs)

            # Handle Result
            if result_state.get("error"):
                # CHANGE THIS LINE
                st.error(f"Error on Image {index + 1}: {result_state['error']}")
            elif result_state.get("structured_data"):
                # Success! Add to list
                q_data = result_state["structured_data"]
                
                # Show mini preview
                with st.expander(f"✅ Question {q_data.id} Processed", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, use_column_width=True)
                    with col2:
                        st.json(q_data.model_dump())

        progress_bar.progress(100)
        status_text.text("Processing Complete!")

# --- Display Final Result ---
if st.session_state.processed_questions:
    st.divider()
    st.subheader("🎉 Final Result")
    
    # Construct the Final JSON using the Container Model
    final_exam = ExamOutput(
        examTitle=exam_title,
        timeLimitMinutes=time_limit,
        questions=st.session_state.processed_questions
    )
    
    # Convert Pydantic model to JSON string
    json_str = final_exam.model_dump_json(indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Preview")
        st.json(json_str)
        
    with col2:
        st.write("### Download")
        st.success(f"Successfully processed {len(st.session_state.processed_questions)} questions.")
        st.download_button(
            label="Download exam.json",
            data=json_str,
            file_name="exam.json",
            mime="application/json"
        )