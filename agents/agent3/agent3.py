"""
Installation Guidance & QA Agent (Vertex AI Version)
Author: AutoFix AI Team
Date: 2025-10-25


This agent uses Vertex AI Gemini via the `google-genai` SDK for reasoning.
It processes live video (OpenCV), audio input (PyAudio), and structured repair steps.
"""


import cv2
import numpy as np
import queue
import threading
import pyaudio
import whisper
import time
from google import genai  # Vertex AI Gemini SDK


# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------


# Environment-based configuration (requires environment variables)
# export GOOGLE_GENAI_USE_VERTEXAI=true
# export GOOGLE_CLOUD_PROJECT="your-project-id"
# export GOOGLE_CLOUD_LOCATION="us-central1"


client = genai.Client()  # Auto-detects Vertex or Gemini mode if vars are set
whisper_model = whisper.load_model("base")


# --------------------------------------------------------
# STEP DATABASE
# --------------------------------------------------------


REPAIR_STEPS = [
    {"id": 1, "action": "Disconnect battery terminal", "tool": "10mm wrench"},
    {"id": 2, "action": "Remove housing bolts", "tool": "12mm socket"},
    {"id": 3, "action": "Clean mounting surface", "tool": "wire brush"},
    {"id": 4, "action": "Install new part", "tool": "13mm socket"},
    {"id": 5, "action": "Reconnect battery", "tool": "none"},
]
# NEED TO CHANGE AGENT REPAIR STEPS IN REFERENCE TO COMPREHENSIVE INSTRUCTION MANUAL
# --------------------------------------------------------
# AUDIO CAPTURE THREAD
# --------------------------------------------------------


def record_audio(q):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=16000,
                     input=True,
                     frames_per_buffer=512)
    while True:
        q.put(stream.read(512))


# --------------------------------------------------------
# SPEECH RECOGNITION / ACOUSTIC SIGNAL
# --------------------------------------------------------


def analyze_audio(queue_audio):
    frames = []
    while not queue_audio.empty():
        frames.append(np.frombuffer(queue_audio.get(), np.int16))
    if not frames:
        return None
    audio_data = np.concatenate(frames)
    result = whisper_model.transcribe(audio_data)
    return result["text"]


# --------------------------------------------------------
# COMPUTER VISION SIMULATION (placeholder logic)
# --------------------------------------------------------


def analyze_video_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    tool_detected = np.sum(mask) > 1000
    return tool_detected


# --------------------------------------------------------
# GEMINI DECISION + FEEDBACK SYSTEM
# --------------------------------------------------------


def generate_feedback(step_id, visual_detected, user_audio_text):
    step = next((s for s in REPAIR_STEPS if s["id"] == step_id), None)
    context = f"""
    Current step: {step['action']}
    Expected tool: {step['tool']}
    Visual detection result: {visual_detected}
    User said: {user_audio_text}
    Provide feedback for a repair AI assistant monitoring a user’s progress.
    Offer safety warnings or confirmations briefly.
    """


    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=context
    )
    return response.text


# --------------------------------------------------------
# MAIN LOOP — Autonomous IGQA Runtime
# --------------------------------------------------------


def run_igqa():
    vid = cv2.VideoCapture(0)
    audio_queue = queue.Queue()
    threading.Thread(target=record_audio, args=(audio_queue,), daemon=True).start()
   
    step_index = 0


    while step_index < len(REPAIR_STEPS):
        ret, frame = vid.read()
        if not ret:
            break


        # Multimodal processing
        visual_detected = analyze_video_frame(frame)
        user_audio = analyze_audio(audio_queue)


        # Reasoning call to Vertex Gemini API
        feedback = generate_feedback(REPAIR_STEPS[step_index]["id"], visual_detected, user_audio)
        print(f"\n🧰 Step {step_index+1} Feedback:")
        print(feedback, "\n")


        # Visual user prompt
        cv2.putText(frame, f"Step {step_index+1}: {REPAIR_STEPS[step_index]['action']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Vertex IGQA Monitor", frame)


        # Manual advancement for prototype
        if cv2.waitKey(10) & 0xFF == ord('n'):
            step_index += 1
       
        time.sleep(2)


    vid.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------
# EXECUTION ENTRY
# --------------------------------------------------------


if __name__ == "__main__":
    print("🦾 Starting Vertex AI IGQA Agent...")
    run_igqa()