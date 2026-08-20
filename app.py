import base64
import io
import json
import os
from PIL import Image
import requests
import streamlit as st

try:
  from gTTS import gTTS

  HAS_GTTS = True
except ImportError:
  HAS_GTTS = False

st.set_page_config(
    page_title="BloggerAgent AI", page_icon="✍️", layout="wide"
)

api_key = (
    st.secrets.get("GEMINI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
).strip()

if not api_key:
  st.error("⚠️ GEMINI_API_KEY not found in Streamlit Secrets!")
  st.stop()


def call_gemini_api(prompt_text, image_obj=None, audio_bytes=None):
  # Standard Google Generative Language Endpoint
  url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
  headers = {
      "Content-Type": "application/json",
      "x-goog-api-key": api_key,
  }

  parts = []

  if image_obj:
    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_str}})

  if audio_bytes:
    aud_str = base64.b64encode(audio_bytes).decode("utf-8")
    parts.append({"inline_data": {"mime_type": "audio/wav", "data": aud_str}})
    parts.append({
        "text": "Transcribe and provide a comprehensive, structured response."
    })
  elif prompt_text:
    parts.append({"text": prompt_text})

  payload = {"contents": [{"parts": parts}]}

  response = requests.post(url, headers=headers, json=payload, timeout=60)
  data = response.json()

  if response.status_code != 200:
    err_detail = data.get("error", {}).get("message", response.text)
    raise Exception(f"{response.status_code}: {err_detail}")

  return data["candidates"][0]["content"]["parts"][0]["text"]


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

if "messages" not in st.session_state:
  st.session_state.messages = []

for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    if msg.get("image"):
      st.image(msg["image"], width=300)
    if msg.get("audio_in"):
      st.audio(msg["audio_in"], format="audio/wav")

    st.markdown(msg["content"])

    if msg.get("audio_out"):
      st.audio(msg["audio_out"], format="audio/mp3")

user_prompt = st.chat_input("Ask anything...")
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

  with st.chat_message("assistant"):
    with st.spinner("Processing..."):
      try:
        ai_text = call_gemini_api(
            prompt_text=user_prompt,
            image_obj=img_data,
            audio_bytes=user_audio_bytes,
        )
      except Exception as e:
        ai_text = f"⚠️ Error: {str(e)}"

      st.markdown(ai_text)

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

  st.session_state.messages.append({
      "role": "assistant",
      "content": ai_text,
      "audio_out": audio_out_bytes,
  })