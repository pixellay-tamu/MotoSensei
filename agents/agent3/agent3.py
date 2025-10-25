# -------------------------------------------------------------------
# igqa_agent.py — Installation Guidance & QA Agent (Vertex AI)
# -------------------------------------------------------------------

"""
Installation Guidance & QA Agent (Vertex AI Version)
Author: AutoFix AI Team
Date: 2025-10-25

This agent:
1. Uses IGGA-generated repair steps as structured guidance.
2. Accepts live user repair video uploads via chatbot.
3. Analyzes video + audio data, giving contextual feedback for each step.
4. Leverages Vertex AI Gemini reasoning for adaptive mechanic coaching.
"""

import os
import json
import tempfile
import cv2
import numpy as np
import queue
import threading
import pyaudio
import whisper
import time
from google import genai
from google.genai import types


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tamu-hackathon25cll-545"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

client = genai.Client()
whisper_model = whisper.load_model("base")


# -------------------------------------------------------------------
# DYNAMIC REPAIR STEP LOADING
# -------------------------------------------------------------------

def load_repair_steps_from_igga(json_steps_path: str):
    """Load the repair guide output from IGGA (JSON)."""
    with open(json_steps_path, "r") as f:
        data = json.load(f)
    return data.get("steps", [])


# -------------------------------------------------------------------
# AUDIO CAPTURE THREAD
# -------------------------------------------------------------------

def record_audio(q):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
    while True:
        q.put(stream.read(512))


# -------------------------------------------------------------------
# SPEECH ANALYSIS (Whisper + Gemini Context)
# -------------------------------------------------------------------

def analyze_audio(queue_audio):
    frames = []
    while not queue_audio.empty():
        frames.append(np.frombuffer(queue_audio.get(), np.int16))
    if not frames:
        return None
    audio_data = np.concatenate(frames)
    result = whisper_model.transcribe(audio_data)
    return result.get("text", "")


# -------------------------------------------------------------------
# VIDEO FRAME ANALYSIS (Tool presence / progress)
# -------------------------------------------------------------------

def analyze_video_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    tool_detected = np.sum(mask) > 1000
    return tool_detected


# -------------------------------------------------------------------
# GEMINI-BASED FEEDBACK DECISION SYSTEM
# -------------------------------------------------------------------

def generate_feedback(step, visual_detected, user_audio_text):
    """Asks Gemini for repair-step-specific feedback given live input."""
    context = f"""
    You are an AI mechanic guiding a human through a repair process.

    Current step: {step['action']}
    Expected tool: {step['tool']}
    Visual tool detected: {visual_detected}
    User commented: {user_audio_text}

    Give concise, conversational feedback:
    - If the user appears to be on track, confirm and encourage.
    - If they're missing a step or using the wrong tool, correct them tactfully.
    - Highlight any safety concerns.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=context
    )
    return response.text


# -------------------------------------------------------------------
# MAIN EXECUTION LOOP
# -------------------------------------------------------------------

def run_igqa(video_source_path: str, repair_steps_path: str):
    """Run the IGQA mechanic agent using an uploaded repair video and IGGA step JSON."""
    print("🧠 Loading guide from IGGA output...")
    repair_steps = load_repair_steps_from_igga(repair_steps_path)
    if not repair_steps:
        print("⚠️ No repair steps found.")
        return

    print("🎥 Opening uploaded video stream...")
    vid = cv2.VideoCapture(video_source_path)
    if not vid.isOpened():
        print("❌ Failed to open video file.")
        return

    audio_queue = queue.Queue()
    threading.Thread(target=record_audio, args=(audio_queue,), daemon=True).start()

    step_index = 0
    print(f"🔧 Starting real-time mechanic guidance for {len(repair_steps)} steps")

    while step_index < len(repair_steps):
        ret, frame = vid.read()
        if not ret:
            break

        visual_detected = analyze_video_frame(frame)
        user_audio = analyze_audio(audio_queue)
        step = repair_steps[step_index]

        feedback = generate_feedback(step, visual_detected, user_audio or "")
        print(f"\n🧰 Step {step_index+1}: {step['action']}")
        print(f"💬 AI Mechanic Feedback:\n{feedback}\n")

        cv2.putText(frame, f"Step {step_index+1}: {step['action']}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.imshow("AutoFix AI Mechanic Monitor", frame)

        if cv2.waitKey(10) & 0xFF == ord('n'):
            step_index += 1

        time.sleep(2)

    vid.release()
    cv2.destroyAllWindows()
    print("✅ Guidance complete.")


# -------------------------------------------------------------------
# EXECUTION EXAMPLE
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example: using the output from IGGA JSON and a user-uploaded video
    igga_guide_path = "latest_repair_guide.json"    # IGGA output file
    uploaded_video_path = "user_repair_video.mp4"   # Chatbot file upload

    print("🦾 Starting Vertex AI IGQA agent...")
    run_igqa(uploaded_video_path, igga_guide_path)
