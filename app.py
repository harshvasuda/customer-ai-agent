import io
import os
from google import genai
from google.genai import types
from PIL import Image
import streamlit as st

# Safe import for gTTS
try:
  from gTTS import gTTS

  HAS_GTTS = True
except ImportError:
  HAS_GTTS = False

st.set_page_config(
    page_title="BloggerAgent AI", page_icon="✍️", layout="wide"
)

# API Key & Client Setup (Auto-detects standard vs OAuth project keys)
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
  st.error("⚠️ GEMINI_API_KEY not found in Streamlit Secrets!")
  st.stop()

# Support both AI Studio keys and OAuth/GCP Tokens
if api_key.startswith("AQ."):
  client = genai.Client(
      api_key=api_key,
      http_options={"api_version": "v1alpha"},
  )
else:
  client = genai.Client(api_key=api_key)

# ----------------- SIDEBAR -----------------
with st.sidebar:
  st.title("✍️ BloggerAgent")
  st.caption("Autonomous Multimodal Content Engine")

  if st.button("➕ New Chat", use_container_width=True, type="primary"):
    st.session_state.messages = []
    st.rerun()

  st.divider()
  st.subheader("📎 Attach Input")
  uploaded_image = st.file_uploader(
      "Upload Image (Optional)", type=["png", "jpg", "jpeg"]
  )
  if uploaded_image:
    st.image(uploaded_image, caption="Attached Image", use_container_width=True)

  voice_audio = st.audio_input("Record Voice Query (Optional)")

# ----------------- MAIN CHAT UI -----------------
st.title("✍️ StreamLite Multimodal Assistant")
st.caption("Research, Draft & Ideate via Text, Voice, or Visual Inputs")

# Initialize Chat Memory
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display Conversation History
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    if msg.get("image"):
      st.image(msg["image"], width=300)
    if msg.get("audio_in"):
      st.audio(msg["audio_in"], format="audio/wav")

    st.markdown(msg["content"])

    if msg.get("audio_out"):
      st.audio(msg["audio_out"], format="audio/mp3")

# Main Chat Bar
user_prompt = st.chat_input(
    "Ask anything... (e.g. Write a tech blog, recipe breakdown, code"
    " explanation)"
)

has_voice_only = voice_audio is not None and not user_prompt
active_input = (
    user_prompt
    if user_prompt
    else ("Voice Note Query" if has_voice_only else None)
)

if active_input:
  img_data = None
  if uploaded_image:
    uploaded_image.seek(0)
    img_data = Image.open(uploaded_image)

  user_audio_bytes = voice_audio.read() if has_voice_only else None

  # 1. User Message
  st.session_state.messages.append({
      "role": "user",
      "content": user_prompt if user_prompt else "🎙️ *[Voice Query Sent]*",
      "image": img_data,
      "audio_in": user_audio_bytes,
  })

  with st.chat_message("user"):
    if img_data:
      st.image(img_data, width=300)
    if user_audio_bytes:
      st.audio(user_audio_bytes, format="audio/wav")
    st.markdown(user_prompt if user_prompt else "🎙️ *[Voice Query Sent]*")

  # 2. Assistant Response
  with st.chat_message("assistant"):
    with st.spinner("Processing & writing..."):
      try:
        payload = []
        if img_data:
          payload.append(img_data)

        if has_voice_only and user_audio_bytes:
          payload.append(
              types.Part.from_bytes(
                  data=user_audio_bytes, mime_type="audio/wav"
              )
          )
          payload.append(
              "Transcribe and provide a comprehensive, structured response in"
              " natural conversational language."
          )
        else:
          payload.append(user_prompt)

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=payload
        )
        ai_text = response.text
      except Exception as e:
        ai_text = f"⚠️ Error: {str(e)}"

      st.markdown(ai_text)

      # Audio Speech Generation
      audio_out_bytes = None
      if HAS_GTTS and not ai_text.startswith("⚠️"):
        try:
          clean_text = (
              ai_text.replace("*", "").replace("#", "").replace("`", "")[:450]
          )
          tts = gTTS(text=clean_text, lang="hi", slow=False)
          fp = io.BytesIO()
          tts.write_to_fp(fp)
          fp.seek(0)
          audio_out_bytes = fp.read()
          st.audio(audio_out_bytes, format="audio/mp3")
        except Exception:
          pass

  # 3. Save Assistant Message
  st.session_state.messages.append({
      "role": "assistant",
      "content": ai_text,
      "audio_out": audio_out_bytes,
  })