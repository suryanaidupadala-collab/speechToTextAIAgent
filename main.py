import streamlit as st
import time
import threading
from datetime import datetime
from queue import Queue, Empty
from audio_recorder import AudioRecorder
from transcriber import TranscriptionWorker

# Page config
st.set_page_config(
    page_title="Speech-to-Text (Faster-Whisper)",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme via CSS
st.markdown("""
<style>
    body { background-color: #121212; color: #e0e0e0; }
    .stButton>button { background-color: #2b2b2b; color: #e0e0e0; border-radius: 6px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #3a3a3a; }
    .stTextArea>textarea { background-color: #1e1e1e; color: #f5f5f5; font-family: Consolas, monospace; }
    .stMarkdown { color: #cfcfcf; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = ''
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'recorder' not in st.session_state:
    st.session_state.recorder = AudioRecorder(samplerate=16000, channels=1, chunk_size=1024)
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = None
if 'status' not in st.session_state:
    st.session_state.status = 'Ready'
if 'text_queue' not in st.session_state:
    st.session_state.text_queue = Queue(maxsize=100)

# Global reference to queue
global_queue = st.session_state.text_queue

# Header
st.title("🎤 Speech-to-Text Transcriber")
st.markdown("Powered by Faster-Whisper")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.text_input("Language Code", value="en", help="e.g., 'en' for English, 'es' for Spanish")
    model_size = st.selectbox("Model Size", ["base", "small", "medium"], help="Larger models are more accurate but slower")
    device = st.radio("Device", ["cpu", "cuda"], help="Use GPU if CUDA-enabled PyTorch is installed")

# Status display
col1, col2 = st.columns([1, 1])
with col1:
    status_metric = st.empty()
    status_metric.metric("Status", st.session_state.status)
with col2:
    time_metric = st.empty()
    if st.session_state.is_recording and st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        m, s = divmod(elapsed, 60)
        time_metric.metric("Elapsed Time", f"{m:02d}:{s:02d}")
    else:
        time_metric.metric("Elapsed Time", "00:00")

# Control buttons
col1, col2, col3, col4, col5 = st.columns(5)

start_clicked = False
with col1:
    if st.button("🎤 Start Speaking", use_container_width=True, key="start_btn"):
        start_clicked = True

with col2:
    if st.button("⏹️ Stop", use_container_width=True, key="stop_btn"):
        if st.session_state.transcriber:
            st.session_state.transcriber.stop()
            time.sleep(0.2)
        try:
            st.session_state.recorder.stop()
        except Exception:
            pass
        st.session_state.is_recording = False
        st.session_state.start_time = None
        st.session_state.status = 'Stopped'
        st.rerun()

with col3:
    if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
        st.session_state.transcript = ''
        st.rerun()

with col4:
    if st.button("📋 Copy", use_container_width=True, key="copy_btn"):
        st.info("Copy text from transcript area with Ctrl+C")

with col5:
    if st.button("💾 Save", use_container_width=True, key="save_btn"):
        if st.session_state.transcript.strip():
            filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.download_button(
                label="Download TXT",
                data=st.session_state.transcript,
                file_name=filename,
                mime="text/plain",
                key="download"
            )
        else:
            st.warning("Transcript is empty")

# Process any queued updates from transcriber
def process_queue():
    items_processed = 0
    while not global_queue.empty() and items_processed < 50:
        try:
            msg_type, content = global_queue.get_nowait()
            if msg_type == 'partial':
                st.session_state.transcript = content
            elif msg_type == 'final':
                st.session_state.transcript += content + ' '
            elif msg_type == 'status':
                st.session_state.status = content
            elif msg_type == 'error':
                st.error(f"Error: {content}")
            items_processed += 1
        except Empty:
            break

# Transcript display placeholder
st.markdown("### 📝 Transcript")
transcript_placeholder = st.empty()

# Handle Start Speaking button
if start_clicked:
    try:
        st.session_state.recorder.start()
        st.session_state.is_recording = True
        st.session_state.start_time = time.time()
        st.session_state.status = 'Loading model...'
        st.session_state.transcript = ''
        
        # Create audio iterator
        audio_iter = st.session_state.recorder.audio_generator()
        
        # Create transcriber
        transcriber = TranscriptionWorker(model_name=model_size, device=device, language=language)
        
        # Thread-safe callbacks
        def on_partial(text):
            try:
                global_queue.put_nowait(('partial', text))
            except:
                pass
        
        def on_final(text):
            try:
                global_queue.put_nowait(('final', text))
            except:
                pass
        
        def on_status(status):
            try:
                global_queue.put_nowait(('status', status))
            except:
                pass
        
        def on_error(e):
            try:
                global_queue.put_nowait(('error', str(e)))
            except:
                pass
        
        transcriber.on_partial = on_partial
        transcriber.on_final = on_final
        transcriber.on_status = on_status
        transcriber.on_error = on_error
        
        st.session_state.transcriber = transcriber
        
        # Start transcriber in background thread
        def run_transcriber():
            try:
                transcriber.start(audio_iter)
            except Exception as e:
                global_queue.put_nowait(('error', str(e)))
        
        thread = threading.Thread(target=run_transcriber, daemon=True)
        thread.start()
        
        # Now enter a live update loop using a placeholder
        with transcript_placeholder.container():
            st.text_area(
                "Your speech will appear here:",
                value=st.session_state.transcript,
                height=300,
                disabled=True,
                label_visibility="collapsed",
                key="transcript_live"
            )
        
        # Poll updates while recording
        poll_iterations = 0
        while st.session_state.is_recording and poll_iterations < 600:  # Max 5 minutes (600 * 0.5s)
            process_queue()
            
            # Update elapsed time
            if st.session_state.start_time:
                elapsed = int(time.time() - st.session_state.start_time)
                m, s = divmod(elapsed, 60)
                time_metric.metric("Elapsed Time", f"{m:02d}:{s:02d}")
            
            status_metric.metric("Status", st.session_state.status)
            
            # Update transcript display
            with transcript_placeholder.container():
                st.text_area(
                    "Your speech will appear here:",
                    value=st.session_state.transcript,
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"transcript_{poll_iterations}"
                )
            
            time.sleep(0.2)
            poll_iterations += 1
        
    except Exception as e:
        st.error(f"Failed to start recorder: {e}")
        st.session_state.is_recording = False
else:
    # Not recording - just show static transcript
    process_queue()
    with transcript_placeholder.container():
        st.text_area(
            "Your speech will appear here:",
            value=st.session_state.transcript,
            height=300,
            disabled=True,
            label_visibility="collapsed",
            key="transcript_display"
        )
