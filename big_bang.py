"""
big_bang.py - FollowMath server

Helps a blind student copy math from a whiteboard onto an iPad.

Two devices share this server:
  - the phone (/camera) watches the whiteboard and streams frames
  - the iPad (/) is where the student draws with a pencil and listens to the tutor

Meta models used:
  - SAM 3.1          locates the equation (and each symbol) in an image
  - Muse Spark 1.3   reads the equation, names symbols, and tutors the student
  - Muse Voice       transcribes spoken questions
"""
import os
import base64
import json
import asyncio
import io
import time
import re
from contextlib import asynccontextmanager
import numpy as np
import cv2
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps
from gtts import gTTS
import subprocess
import tempfile
import imageio_ffmpeg
from dotenv import load_dotenv
from symbol_map import build_symbol_map

load_dotenv()
META_API_KEY = os.getenv("META_API_KEY")
META_CHAT_ENDPOINT = "https://api.meta.ai/v1/chat/completions"
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except (ImportError, RuntimeError):
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# --- Shared state ---
pipeline_data = {
    "x1": 0, "y1": 0, "x2": 0, "y2": 0,
    "start_x": 0, "start_y": 0,
    "width": 0,
    "semantic_text": "",
    "zones": {},
    "raw_stencil_mask": None,
    "ready": False,   # True once an equation has been read from the board
}

audio_cache = {}      # text -> base64 mp3

# Connection hub: the phone and the iPad talk through the server
hub = {
    "ipad_ws": None,
    "phone_ws": None,
    "inbox": None,      # the iPad session's message queue; camera frames from the phone are put here
    "live_on": False,   # whether the iPad wants live frames from the phone
}

async def push_phone_live():
    # Tell the phone whether to stream live frames
    phone = hub["phone_ws"]
    if phone is not None:
        try:
            await phone.send_json({"action": "live", "on": hub["live_on"]})
        except Exception:
            pass

async def notify_ipad(obj):
    ipad = hub["ipad_ws"]
    if ipad is not None:
        try:
            await ipad.send_json(obj)
        except Exception:
            pass

async def notify_phone(obj):
    phone = hub["phone_ws"]
    if phone is not None:
        try:
            await phone.send_json(obj)
        except Exception:
            pass

async def get_fast_voice(text):
    # Text-to-speech with an in-memory cache. Returns base64 mp3, or None on failure.
    if text in audio_cache:
        return audio_cache[text]

    def sync_tts():
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode('utf-8')

    try:
        b64_audio = await asyncio.to_thread(sync_tts)
        audio_cache[text] = b64_audio
        return b64_audio
    except Exception as e:
        print(f"TTS failed: {e}")
        return None

# --- 1. Meta API calls ---
async def call_sam_3_1(image_path: str):
    # Ask SAM 3.1 for the bounding box of the equation. Returns (x1, y1, x2, y2) or None.
    print("1. Calling SAM 3.1 to locate the equation...")
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {META_API_KEY}",
        "Content-Type": "application/json"
    }

    sam_payload = {
        "model": "sam-3.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "math equation"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"}
                ]
            }
        ]
    }

    sam_endpoint = "https://api.meta.ai/v1/responses"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(sam_endpoint, headers=headers, json=sam_payload, timeout=15.0)
            if response.status_code == 200:
                result = response.json()
                sam_text = result["output"][0]["content"][0]["text"]
                box_match = re.search(r"<\|box;x1=(\d+(?:\.\d+)?);y1=(\d+(?:\.\d+)?);x2=(\d+(?:\.\d+)?);y2=(\d+(?:\.\d+)?);", sam_text)
                if box_match:
                    x1, y1, x2, y2 = (int(float(v)) for v in box_match.groups())
                    print(f"Bounding box found: ({x1}, {y1}) to ({x2}, {y2})")
                    return x1, y1, x2, y2
                print(f"   SAM answered but gave no box. It said: {sam_text[:200]!r}")
            else:
                print(f"   SAM HTTP {response.status_code}: {response.text[:200]}")
        except Exception as e:
            print(f"SAM 3.1 connection failed: {type(e).__name__}: {e!r}")

    return None   # the caller falls back to the whole photo

async def call_muse_spark(image_path: str, bbox: tuple):
    # Have Muse Spark read the cropped equation and split it into spoken left/center/right thirds.
    print("2. Asking Muse Spark to read the equation...")
    img = Image.open(image_path)
    cropped_img = img.crop(bbox)

    buffered = io.BytesIO()
    cropped_img.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {META_API_KEY}",
        "Content-Type": "application/json"
    }

    # Structured JSON output so each third of the equation can be spoken on its own
    payload = {
        "model": "muse-spark-1.3",
        "messages": [
            {
                "role": "system",
                "content": "You are a math accessibility assistant helping a blind student. Look at the picture of a written math equation (handwritten or on a whiteboard). Output ONLY a valid JSON object with exactly these 5 keys: 'found' (true only if a written math equation is clearly visible, otherwise false), 'full_audio' (the entire equation spoken in English), 'left_audio' (the spoken English for the left third of the equation), 'center_audio' (the spoken English for the middle third), 'right_audio' (the spoken English for the right third)."
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(META_CHAT_ENDPOINT, headers=headers, json=payload, timeout=15.0)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # Remove markdown code fences if the model wraps its JSON in them
                content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)
        except Exception as e:
            print(f"Spark parsing failed: {e}")

    # No placeholder equation on failure: the caller asks the user to retake the picture
    return None

async def transcribe_audio(audio_b64: str):
    # Convert the recorded question to WAV and transcribe it with Muse Voice
    print("5. Transcribing the student's question...")

    if "," in audio_b64:
        audio_b64 = audio_b64.split(",")[1]

    audio_data = base64.b64decode(audio_b64)

    # Ignore empty recordings (e.g. the button was tapped twice quickly)
    if len(audio_data) < 100:
        print("Audio payload was empty.")
        return ""

    # iOS records MP4; the API needs mono 24 kHz PCM WAV
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
            temp_in.write(audio_data)
            temp_in_path = temp_in.name

        temp_out_path = temp_in_path.replace(".mp4", ".wav")

        subprocess.run([
            FFMPEG_PATH, "-y", "-i", temp_in_path,
            "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", "-map_metadata", "-1",
            temp_out_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(temp_out_path, "rb") as f:
            wav_data = f.read()

        os.remove(temp_in_path)
        os.remove(temp_out_path)

    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        return ""

    data = {
        "request": json.dumps({
            "model": "muse-voice-transcribe-1.0",
            "mode": "PUSH_TO_TALK",
            "audioEncoding": "WAV"
        })
    }

    files = {
        'audio': ('question.wav', wav_data, 'audio/wav')
    }

    headers = {"Authorization": f"Bearer {META_API_KEY}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.meta.ai/v1/asr/transcribe",
                headers=headers,
                data=data,
                files=files,
                timeout=15.0
            )
            if response.status_code == 200:
                transcript = response.json().get("transcript", "")
                print(f"User asked: {transcript}")
                return transcript
            else:
                print(f"Voice API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Voice transcription failed: {e}")
    return ""

async def evaluate_canvas_with_spark(b64_image_data: str, target_equation: str):
    # Tutor feedback on the student's drawing, written as a short script for text-to-speech
    print("4. Grading the canvas with Muse Spark...")
    if b64_image_data.startswith("data:image"):
        b64_image_data = b64_image_data.split(",")[1]

    headers = {
        "Authorization": f"Bearer {META_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "muse-spark-1.3",
        "messages": [
            {
                "role": "system",
                "content": f"You are a supportive math tutor for a blind student. The target equation is: '{target_equation}'. Look at their current drawing. 1. Acknowledge what they have drawn correctly. 2. If they made a mistake, gently correct them. 3. Only confirm a mathematical symbol is correct if it is fully and clearly drawn. If it looks like a partial or unfinished stroke, tell the student to finish drawing it 4. CRITICAL: If their drawing is correct up to this point, you MUST explicitly tell them the EXACT NEXT mathematical symbol they need to draw to continue. 5. If a symbol is missing or incomplete (for example an equals sign with only one line), say exactly which symbol is incomplete. Respond ONLY with the exact spoken English script for the text-to-speech engine in 1 to 3 short sentences."
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image_data}"}}]
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        for attempt in (1, 2):
            try:
                response = await client.post(META_CHAT_ENDPOINT, headers=headers, json=payload, timeout=25.0)
                if response.status_code == 200:
                    verdict = response.json()["choices"][0]["message"]["content"].strip()
                    print(f"Tutor feedback: {verdict}")
                    return verdict
                print(f"Evaluation attempt {attempt}: HTTP {response.status_code}: {response.text[:300]}")
            except Exception as e:
                print(f"Evaluation attempt {attempt} failed: {type(e).__name__}: {e!r}")
            await asyncio.sleep(1.5)   # brief pause before the single retry
    return "I am having trouble seeing your drawing right now, but keep going."

# --- 2. Scan pipeline: camera photo -> equation -> stencil ---
SCAN_INTRO = (
    "Welcome to FollowMath. Prop the phone up so its back camera sees the whiteboard, "
    "then tap anywhere on the phone screen and hold still until you hear the shutter sound. "
    "You only do this once. After that I watch the board and tell you when the professor writes."
)
SCAN_WAIT = "Got it. Reading the equation, one moment."
MAP_WAIT = "Checking what you wrote, one moment."
NOTHING_HERE = "Nothing here."
NOTHING_LEFT = "Nothing here. Try moving left."
NOTHING_RIGHT = "Nothing here. Try moving right."
MAP_EMPTY = "I don't see anything written yet."
WAITING_MSG = ("Board position saved. I'm watching the board now. I'll tell you as soon as the professor "
               "writes something, and keep you updated as the equation grows.")
NOT_READY_MSG = "The board is still empty. I'll tell you as soon as something is written."
PHONE_MISSING = "The phone camera is not connected. Open the camera page on the phone, then tap again."
PHONE_TIMEOUT = "The phone camera did not answer. Check that the phone screen is on, then tap to try again."
PHONE_ON = "Phone camera connected."
PHONE_OFF = "Phone camera disconnected."
CAMERA_ERROR_MSG = ("I couldn't open the phone camera. Please allow camera access in the phone's browser "
                    "settings, then reload the camera page on the phone.")
NORMALIZED_WIDTH = 520   # scanned equations are resized to this width so navigation thresholds stay consistent
MAX_PHOTO_SIDE = 1280    # photos are downscaled before being sent to the models

# --- Live mode: the phone keeps watching the board while the student works ---
LIVE_INK_CHANGE = 0.06   # below 6% change in visible ink, skip the model calls

def ink_signature(img):
    # Cheap fingerprint of the board: the amount of ink in view
    g = img.convert("L")
    g = g.resize((320, max(1, int(g.height * 320 / g.width))))
    g = cv2.GaussianBlur(np.array(g), (3, 3), 0)
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    return int((ink > 0).sum())

def _norm_text(t):
    # Normalize for comparing two readings of the equation
    return re.sub(r"[^a-z0-9]+", "", str(t).lower())

def _waiting_state(img):
    # Setup only positions the camera. An empty board is expected; live mode announces
    # the equation as soon as the professor starts writing.
    pipeline_data.update({
        "x1": 0, "y1": 0, "x2": 0, "y2": 0, "start_x": 0, "start_y": 0, "width": 0,
        "semantic_text": "", "zones": {},
        "raw_stencil_mask": np.zeros((1, 1), np.uint8),
        "ready": False,
        "live_sig": ink_signature(img),
    })
    return {"ok": True, "waiting": True, "changed": False, "message": WAITING_MSG}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-generate common prompts so they play without delay
    print("Pre-generating common voice prompts...")
    for phrase in (SCAN_INTRO, SCAN_WAIT, "Canvas cleared. Let's start over.", "Undone.",
                   "Thinking...", "Let me check what you have written so far...",
                   "I didn't quite catch that.", "Equation anchored! Start writing.",
                   MAP_WAIT, NOTHING_HERE, NOTHING_LEFT, NOTHING_RIGHT, MAP_EMPTY,
                   WAITING_MSG, NOT_READY_MSG, PHONE_MISSING, PHONE_TIMEOUT, PHONE_ON, PHONE_OFF,
                   CAMERA_ERROR_MSG):
        await get_fast_voice(phrase)
    print("Server ready. Open the page and scan a whiteboard.")
    yield

async def process_scan(image_b64: str, live: bool = False):
    # Find and read the equation in a camera photo and build the ink stencil.
    # Returns {"ok": bool, "message": str, ...}
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        print(f"Scan decode failed: {e}")
        return {"ok": False, "message": "I couldn't read that picture. Please tap to try again."}

    longest = max(img.size)
    if longest > MAX_PHOTO_SIDE:
        ratio = MAX_PHOTO_SIDE / longest
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    W, H = img.size

    if live:
        # Skip the model calls when the amount of ink barely changed
        sig = ink_signature(img)
        prev = pipeline_data.get("live_sig")
        if prev and abs(sig - prev) / max(prev, 1) < LIVE_INK_CHANGE:
            print(f"   Live frame: ink changed {abs(sig - prev) / max(prev, 1):.1%} (< {LIVE_INK_CHANGE:.0%}), skipped")
            return {"ok": True, "changed": False}
        print(f"   Live frame: ink changed {abs(sig - (prev or sig)) / max(prev or sig, 1):.1%}, reading the board...")
        pipeline_data["live_sig"] = sig

    scan_path = os.path.join(tempfile.gettempdir(), "followmath_scan.png")
    img.save(scan_path, format="PNG")

    # Step 1: locate the equation
    full_spark = None
    used_full_frame = False
    if live:
        # For live frames, start Spark on the full frame in parallel with SAM so a slow or
        # failed SAM call never delays an update.
        full_spark = asyncio.create_task(call_muse_spark(scan_path, (0, 0, W, H)))
    bbox = await call_sam_3_1(scan_path)
    if bbox is None and live:
        print("   Live frame: SAM found no box, using Spark's read of the full frame")
        bbox = (0, 0, W, H)
        used_full_frame = True
    if bbox is None:
        print("SAM found no equation, waiting for the professor to start writing.")
        return _waiting_state(img)
    x1, x2 = max(0, min(bbox[0], bbox[2])), min(W, max(bbox[0], bbox[2]))
    y1, y2 = max(0, min(bbox[1], bbox[3])), min(H, max(bbox[1], bbox[3]))
    if (x2 - x1) < 40 or (y2 - y1) < 20:
        x1, y1, x2, y2 = 0, 0, W, H     # box too small to be useful: use the whole frame
    bbox = (x1, y1, x2, y2)
    print(f"Using crop {bbox} of a {W}x{H} photo")

    # Step 2: read it
    if full_spark is not None and used_full_frame:
        semantic_data = await full_spark
    else:
        if full_spark is not None:
            full_spark.cancel()          # SAM found the equation, so read just the crop
        semantic_data = await call_muse_spark(scan_path, bbox)
    if live:
        if isinstance(semantic_data, dict):
            print(f"   Live frame: Spark found={semantic_data.get('found')} text={str(semantic_data.get('full_audio', ''))[:80]!r}")
        else:
            print("   Live frame: Spark gave no usable answer, will retry on the next frame")
    if isinstance(semantic_data, dict) and semantic_data.get("found") is False:
        if live:
            return {"ok": False}
        return _waiting_state(img)
    if not isinstance(semantic_data, dict) or not str(semantic_data.get("full_audio", "")).strip():
        if live and prev:
            pipeline_data["live_sig"] = prev     # read failed: don't mark this view as seen
        return {"ok": False, "message": "I had trouble reading the picture. Please tap to try again."}

    if live:
        pipeline_data["live_sig"] = sig
        if _norm_text(semantic_data["full_audio"]) == _norm_text(pipeline_data["semantic_text"]):
            print("   Live frame: equation reads the same, staying silent")
            return {"ok": True, "changed": False}

    # Step 3: build the ink stencil (normalized size, adaptive threshold for uneven lighting)
    color_crop = img.crop(bbox)
    norm_h = max(20, int(color_crop.height * NORMALIZED_WIDTH / color_crop.width))
    color_crop = color_crop.resize((NORMALIZED_WIDTH, norm_h), Image.LANCZOS)

    gray = cv2.GaussianBlur(np.array(color_crop.convert("L")), (3, 3), 0)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)

    # Remove small specks (dust, marker residue, sensor noise)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 12
    keep[0] = False
    raw_mask = keep[labels].astype(np.uint8)

    if int(raw_mask.sum()) < 150:
        return {"ok": False, "message": "I couldn't make out any writing. Hold the camera steady and a little closer, then tap to try again."}

    ink_y, ink_x = np.where(raw_mask == 1)
    full_audio = str(semantic_data["full_audio"]).strip()

    pipeline_data.update({
        "x1": 0, "y1": 0, "x2": NORMALIZED_WIDTH, "y2": norm_h,
        "width": NORMALIZED_WIDTH,
        "raw_stencil_mask": raw_mask,
        "start_x": int(np.min(ink_x)),
        "start_y": int(ink_y[np.argmin(ink_x)]),
        "semantic_text": full_audio,
        "zones": semantic_data,
        "ready": True,
    })

    buf = io.BytesIO()
    color_crop.save(buf, format="PNG")
    crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    intro_msg = (f"Equation loaded. {full_audio} Tap anywhere on the screen to start drawing. "
                 "Draw with your pencil. To hear what you have written, slide one finger over your writing and I will name each symbol. "
                 "Use the microphone button at the bottom right to ask me a question, "
                 "or say scan a new equation to take another picture.")
    pipeline_data["live_sig"] = ink_signature(img)
    return {"ok": True, "changed": True, "message": intro_msg, "crop_b64": crop_b64, "semantic_text": full_audio}

app = FastAPI(lifespan=lifespan)

# --- 3. Navigation: left -> middle -> right gates ---
def get_smart_navigation(current_x, current_y, active_stencil):
    # Returns a spoken correction, or None when the student is on track.
    remaining_y, remaining_x = np.where(active_stencil > 0)

    # Done when fewer than 50 stencil pixels remain anywhere
    if len(remaining_x) < 50:
        return "Equation complete! Excellent job."

    bound_1 = pipeline_data["width"] * 0.33
    bound_2 = pipeline_data["width"] * 0.66
    zones = pipeline_data["zones"]

    # Count the stencil pixels still uncovered in each zone
    left_remaining_count = np.sum(remaining_x < bound_1)
    center_remaining_count = np.sum((remaining_x >= bound_1) & (remaining_x < bound_2))

    # Stay quiet unless the student drifts off track.

    # Left third not finished yet
    if left_remaining_count > 150:
        if current_x > bound_1 + 40:
            return f"You drifted too far right. Go back left to finish: {zones.get('left_audio', 'the first part')}."
        return None

    # Middle third
    elif center_remaining_count > 150:
        if current_x < bound_1 - 40:
            return "Move right to reach the middle of the equation."
        if current_x > bound_2 + 40:
            return "You drifted too far right. Go back left to the middle of the equation."
        return None

    # Right third
    else:
        if current_x < bound_2 - 40:
            return "Move right to finish the equation."
        return None

# --- 4. iPad frontend ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>FollowMath AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { background: #1e1e2e; color: white; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; margin: 0; padding: 15px; height: 100vh; box-sizing: border-box; overscroll-behavior: none; overflow: hidden; }
        #start-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #11111b; z-index: 9999; display: flex; align-items: center; justify-content: center; color: #a6e3a1; font-size: 28px; font-weight: bold; text-align: center; padding: 20px; cursor: pointer; touch-action: manipulation; }
        #reference-img { max-width: 90%; max-height: 80px; border-radius: 8px; margin-bottom: 5px; border: 2px solid #555; }
        .muse-translation { background: #181825; padding: 8px 20px; border-radius: 6px; color: #a6adc8; font-size: 14px; margin-bottom: 15px; text-align: center; max-width: 95%; font-style: italic; border-left: 4px solid #cba6f7; }
        #canvas-container { width: 95%; flex-grow: 1; position: relative; }
        canvas { width: 100%; height: 100%; background: #2a2a3b; border: 2px dashed #6c7086; border-radius: 8px; touch-action: none; cursor: crosshair; user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; }
        /* Large touch target pinned to the bottom right */
        #mic-btn { position: fixed; bottom: 20px; right: 20px; background: #cba6f7; color: #11111b; border: none; border-radius: 16px; width: 72px; height: 72px; font-size: 32px; box-shadow: 0 6px 12px rgba(0,0,0,0.5); cursor: pointer; display: flex; align-items: center; justify-content: center; touch-action: manipulation; transition: transform 0.2s; z-index: 999; }
        #mic-btn:active { transform: scale(0.92); }
        #mic-btn.recording { background: #f38ba8; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(243, 139, 168, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(243, 139, 168, 0); } 100% { box-shadow: 0 0 0 0 rgba(243, 139, 168, 0); } }

        /* Side padding keeps the status text clear of the mic button */
        #status-box { margin-top: 15px; padding: 15px 100px; background: #313244; border-radius: 8px; width: 95%; text-align: center; font-size: 20px; font-weight: bold; min-height: 60px; box-sizing: border-box; margin-bottom: 20px; }
        .hidden { display: none !important; }
        /* Camera stays active but invisible so the board can be re-read while the student draws */
        #scanner-view.bg-camera { width: 4px; height: 4px; opacity: 0; pointer-events: none; z-index: -1; overflow: hidden; }
        #scanner-view { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #000; z-index: 500; touch-action: manipulation; }
        #cam-video { width: 100%; height: 100%; object-fit: cover; }
        #scan-hint { position: absolute; left: 0; right: 0; bottom: 0; padding: 40px 20px 50px; background: linear-gradient(transparent, rgba(0,0,0,0.85)); color: #fff; font-size: 28px; font-weight: bold; text-align: center; }
        #draw-view { width: 100%; flex-grow: 1; display: flex; flex-direction: column; align-items: center; min-height: 0; }
        #cam-btn { position: fixed; bottom: 20px; left: 20px; background: #89b4fa; color: #11111b; border: none; border-radius: 16px; width: 72px; height: 72px; font-size: 32px; box-shadow: 0 6px 12px rgba(0,0,0,0.5); cursor: pointer; display: flex; align-items: center; justify-content: center; touch-action: manipulation; z-index: 999; }
        #cam-btn:active { transform: scale(0.92); }
        #cam-btn { display: none !important; }   /* hidden: rescanning is done by voice ("scan a new equation") */
    </style>
</head>
<body>
    <div id="start-overlay" role="button" aria-label="Start FollowMath AI. Tap anywhere.">Tap anywhere to start FollowMath AI</div>

    <!-- One persistent audio element; Safari drops queued audio when elements are recreated -->
    <audio id="voice-player" playsinline></audio>

    <!-- Scanner view: full-screen, tap anywhere to take the picture -->
    <div id="scanner-view" class="hidden" role="button" aria-label="Tap anywhere to take a picture of the whiteboard with the phone camera.">
        <div id="scan-hint">Tap anywhere to take a picture</div>
    </div>

    <!-- Drawing view: shown after a successful scan -->
    <div id="draw-view" class="hidden">
        <img id="reference-img" alt="Scanned equation">
        <div class="muse-translation">Muse Spark Translation: "<span id="semantic-text"></span>"</div>
        <div id="canvas-container">
            <canvas id="drawingCanvas" width="800" height="300"></canvas>
        </div>
    </div>

    <div id="status-box">Connecting...</div>
    <button id="cam-btn" class="hidden" aria-label="Scan a new equation">📷</button>
    <button id="mic-btn" class="hidden" aria-label="Ask the tutor a question">🎙️</button>

    <script>
        const canvas = document.getElementById("drawingCanvas");
        const container = document.getElementById("canvas-container");
        const ctx = canvas.getContext("2d");
        const statusBox = document.getElementById("status-box");
        const startOverlay = document.getElementById("start-overlay");
        const voicePlayer = document.getElementById("voice-player");
        const scannerView = document.getElementById("scanner-view");
        let phoneConnected = false;     // whether the phone camera page is connected
        let scanTimeout = null;
        let liveWanted = false;
        const scanHint = document.getElementById("scan-hint");
        const drawView = document.getElementById("draw-view");
        const refImg = document.getElementById("reference-img");

        voicePlayer.playbackRate = 1.35;

        function applyInkStyle() {
            ctx.lineWidth = 4;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.strokeStyle = "#a6e3a1";
        }

        function resizeCanvas(force) {
            if (container.clientWidth === 0) return;   // container is hidden
            if (!force && canvas.width === container.clientWidth && canvas.height === container.clientHeight) return;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            applyInkStyle();
        }
        window.addEventListener('resize', () => resizeCanvas(false));
        applyInkStyle();

        let isDrawing = false;
        let chunkPath = [];
        let currentStroke = [];
        let strokeHistory = [];
        let lastHoverTime = 0;

        let audioQueue = [];
        let isPlaying = false;
        let evaluationTimeout;
        let pendingStrokes = false;

        let mediaRecorder = null;
        let micStream = null;
        let audioChunks = [];
        let isRecordingHelp = false;
        let micStarting = false;

        // ---------- Touch explorer: finger = hear symbols, pencil = draw ----------
        let strokeVersion = 0;            // incremented whenever the drawing changes
        let symbolMap = { version: -1, segments: [], nothing: null, left: null, right: null };
        let touchActiveId = null;
        let lastSymbolIndex = -2;
        let lastExploreTime = 0;
        let lastMapRequest = 0;
        let lastPenTime = 0;
        function markCanvasChanged() { strokeVersion++; }
        let boardReady = false;           // false until an equation has been read from the board

        let camStream = null;
        let scanBusy = false;
        let mode = "start";      // start | scan | draw
        let audioCtx = null;

        async function startRecording() {
            // Open a fresh mic stream each time: Safari mutes long-lived streams once playback starts
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            mediaRecorder = new MediaRecorder(micStream);

            mediaRecorder.ondataavailable = e => {
                if (e.data && e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const chunks = audioChunks;
                audioChunks = [];

                // Release the mic so playback routing returns to normal
                if (micStream) {
                    micStream.getTracks().forEach(t => t.stop());
                    micStream = null;
                }

                if (chunks.length === 0) {
                    statusBox.innerText = "Error: 0 bytes recorded. Try again.";
                    return;
                }

                statusBox.innerText = "Sending audio to server...";
                const audioBlob = new Blob(chunks);
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64Audio = reader.result;
                    const canvasData = getSnapshot();
                    ws.send(JSON.stringify({ action: "ask_tutor", audio_b64: base64Audio, image_b64: canvasData }));
                };
            };

            mediaRecorder.start();
        }

        voicePlayer.onended = () => { playNextAudio(); };

        function playNextAudio() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }
            isPlaying = true;
            const nextB64 = audioQueue.shift();
            voicePlayer.src = "data:audio/mp3;base64," + nextB64;
            voicePlayer.play().catch((err) => {
                console.error("Audio playback error:", err);
                playNextAudio();
            });
        }

        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(wsProtocol + "//" + window.location.host + "/ws");

        startOverlay.addEventListener("pointerdown", () => {
            startOverlay.style.display = "none";

            // iOS only allows audio after a user gesture, so unlock playback on the first tap
            voicePlayer.src = "data:audio/mp3;base64,";
            voicePlayer.play().catch(() => {});
            try {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                audioCtx.resume();
            } catch (e) {}

            enterDrawMode();
        });

        const micBtn = document.getElementById("mic-btn");
        const camBtn = document.getElementById("cam-btn");

        micBtn.addEventListener("pointerdown", async (e) => {
            e.preventDefault();
            if (micStarting || mode !== "draw") return;

            // Stop recording
            if (mediaRecorder && mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                isRecordingHelp = false;
                pushLive();
                micBtn.classList.remove("recording");
                return;
            }

            // Start recording
            micStarting = true;
            isRecordingHelp = true;
            pushLive();

            clearTimeout(evaluationTimeout);
            pendingStrokes = false;
            voicePlayer.pause();
            audioQueue = [];
            isPlaying = false;

            try {
                await startRecording();
                micBtn.classList.add("recording");
                statusBox.innerText = "Listening... Tap mic again to send.";
            } catch (err) {
                console.error("Mic error:", err);
                isRecordingHelp = false;
                pushLive();
                statusBox.innerText = "Microphone error: " + err;
            } finally {
                micStarting = false;
            }
        });

        // ---------- Whiteboard scanning ----------
        function setStatus(t) {
            statusBox.innerText = t;
            scanHint.innerText = t;
        }

        function sendWS(obj) {
            if (ws.readyState === 1) ws.send(JSON.stringify(obj));
            else ws.addEventListener("open", () => ws.send(JSON.stringify(obj)), { once: true });
        }

        function beep(freq, dur) {
            try {
                if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (audioCtx.state === "suspended") audioCtx.resume();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.frequency.value = freq;
                gain.gain.value = 0.25;
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + dur);
            } catch (e) { console.error("Beep failed:", e); }
        }

        // The camera lives on the phone (/camera). The iPad never opens a camera.

        function enterDrawMode() {
            // Go straight to the drawing screen. The whiteboard picture is taken on the phone;
            // the result arrives here over the websocket.
            mode = "draw";
            scanBusy = false;
            boardReady = false;
            scannerView.classList.add("hidden");
            drawView.classList.remove("hidden");
            micBtn.classList.remove("hidden");
            camBtn.classList.remove("hidden");
            refImg.style.display = "none";
            document.querySelector(".muse-translation").style.display = "none";
            resizeCanvas(true);
            setStatus("Waiting for the first picture from the phone.");

            // The server speaks the instructions
            sendWS({ action: "request_intro" });
        }

        function requestRescan() {
            // Ask the phone to run its countdown beeps and take a new picture
            if (scanBusy) return;
            scanBusy = true;
            setStatus("Asking the phone to take a new picture...");
            sendWS({ action: "scan_request" });
            clearTimeout(scanTimeout);
            scanTimeout = setTimeout(() => {
                if (scanBusy) {
                    scanBusy = false;
                    setStatus("The phone camera did not answer. Try again.");
                    sendWS({ action: "scan_timeout" });
                }
            }, 25000);
        }

        camBtn.addEventListener("pointerdown", (e) => {
            e.preventDefault();
            if (mediaRecorder && mediaRecorder.state === "recording") return;
            voicePlayer.pause();
            audioQueue = [];
            isPlaying = false;
            requestRescan();
        });

        // ---------- Live mode: the board is re-read every few seconds ----------
        const LIVE_INTERVAL_MS = 8000;
        let liveTimer = null;

        // The phone owns the timer and sends the frames. The iPad only says whether it wants them
        // (drawing mode, and not while the student is recording a question).
        function pushLive() {
            sendWS({ action: "live_control", on: liveWanted && !isRecordingHelp });
        }

        function startLive() {
            liveWanted = true;
            pushLive();
        }

        function stopLive() {
            liveWanted = false;
            pushLive();
        }

        function redrawStrokes() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            strokeHistory.forEach(stroke => {
                if (stroke.length < 2) return;
                ctx.beginPath();
                ctx.moveTo(stroke[0][0], stroke[0][1]);
                for (let i = 1; i < stroke.length; i++) ctx.lineTo(stroke[i][0], stroke[i][1]);
                ctx.stroke();
            });
            ctx.beginPath();
        }

        function applyLiveUpdate(r) {
            // The professor wrote more: refresh the reference image and text, keep the student's drawing
            boardReady = true;
            refImg.style.display = "";
            document.querySelector(".muse-translation").style.display = "";
            document.getElementById("semantic-text").innerText = r.semantic_text;
            refImg.onload = () => { resizeCanvas(false); redrawStrokes(); };
            refImg.src = "data:image/png;base64," + r.crop_b64;
        }

        function applyScan(r) {
            if (!r.ok) {
                // The camera keeps running, so the user can simply tap again
                scanBusy = false;
                scanHint.innerText = "Tap anywhere to try again";
                return;
            }
            // Camera stays on: live mode keeps watching the board in the background
            mode = "draw";
            scanBusy = false;
            scannerView.classList.add("hidden");
            startLive();
            drawView.classList.remove("hidden");
            micBtn.classList.remove("hidden");
            camBtn.classList.remove("hidden");

            document.getElementById("semantic-text").innerText = r.semantic_text;
            boardReady = !r.waiting;
            refImg.style.display = r.waiting ? "none" : "";
            document.querySelector(".muse-translation").style.display = r.waiting ? "none" : "";
            refImg.onload = () => resizeCanvas(true);
            refImg.src = "data:image/png;base64," + r.crop_b64;

            strokeHistory = [];
            chunkPath = [];
            currentStroke = [];
            markCanvasChanged();
            resizeCanvas(true);
        }

        function clearCanvas() {
            strokeHistory = [];
            chunkPath = [];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            markCanvasChanged();
            ws.send(JSON.stringify({ action: "clear_canvas" }));
        }

        function undoLastStroke() {
            if (strokeHistory.length === 0) {
                // Let the server announce it, since the user can't see the screen
                ws.send(JSON.stringify({ action: "undo_canvas", history: [], nothing_to_undo: true }));
                return;
            }
            strokeHistory.pop();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();

            strokeHistory.forEach(stroke => {
                if(stroke.length > 0) {
                    ctx.moveTo(stroke[0][0], stroke[0][1]);
                    for(let i=1; i<stroke.length; i++) ctx.lineTo(stroke[i][0], stroke[i][1]);
                    ctx.stroke();
                }
            });
            markCanvasChanged();
            ws.send(JSON.stringify({ action: "undo_canvas", history: strokeHistory }));
        }

        ws.onopen = () => { statusBox.innerText = "Ready. Tap the screen to begin."; };
        ws.onclose = () => { setStatus("Connection lost. Please reload the page."); };

        ws.onmessage = (event) => {
            const response = JSON.parse(event.data);

            if (response.action === "force_undo") {
                undoLastStroke();
                return;
            }
            if (response.action === "force_clear") {
                clearCanvas();
                return;
            }
            if (response.action === "force_rescan") {
                requestRescan();
                return;
            }
            if (response.action === "request_snapshot") {
                if (isRecordingHelp) return;   // don't evaluate while the user is speaking
                const canvasData = getSnapshot();
                ws.send(JSON.stringify({ action: "evaluate", image_b64: canvasData, version: strokeVersion }));
                pendingStrokes = false;
                return;
            }
            if (response.action === "symbol_map") {
                symbolMap = {
                    version: response.version,
                    segments: response.segments || [],
                    nothing: response.nothing_audio,
                    left: response.left_audio,
                    right: response.right_audio
                };
                return;
            }
            if (response.action === "symbol_map_failed") return;
            if (response.action === "live_update") applyLiveUpdate(response);   // falls through so the message is spoken
            if (response.action === "phone_status") phoneConnected = !!response.connected;
            if (response.action === "scan_result") { clearTimeout(scanTimeout); applyScan(response); }

            // Ignore tutor audio while the mic is open
            if (isRecordingHelp && response.audio_b64) return;

            if (response.message) setStatus(response.message);

            if (response.audio_b64) {
                if (response.priority) {
                    audioQueue = [response.audio_b64];
                    playNextAudio();
                } else {
                    audioQueue = [response.audio_b64];
                    if (!isPlaying) playNextAudio();
                }
            }
        };

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
        }

        // Export a high-contrast image for the models: black ink on white, thicker lines, 2x size.
        // The raw canvas (thin green ink on transparent) is often misread, especially thin symbols like "=".
        function getSnapshot() {
            const k = 2;
            const out = document.createElement("canvas");
            out.width = canvas.width * k;
            out.height = canvas.height * k;
            const o = out.getContext("2d");
            o.fillStyle = "#ffffff";
            o.fillRect(0, 0, out.width, out.height);
            o.strokeStyle = "#000000";
            o.lineWidth = 6;
            o.lineCap = "round";
            o.lineJoin = "round";
            const all = isDrawing ? strokeHistory.concat([currentStroke]) : strokeHistory;
            all.forEach(stroke => {
                if (stroke.length < 2) return;
                o.beginPath();
                o.moveTo(stroke[0][0] * k, stroke[0][1] * k);
                for (let i = 1; i < stroke.length; i++) o.lineTo(stroke[i][0] * k, stroke[i][1] * k);
                o.stroke();
            });
            return out.toDataURL("image/png");
        }

        // ---------- Finger explorer ----------
        // Find the symbol under the finger. Boxes are padded because a fingertip is imprecise.
        function symbolAt(pos) {
            const W = canvas.width, H = canvas.height, pad = 22;
            let best = -1, bestDist = Infinity;
            symbolMap.segments.forEach((s, i) => {
                const [fx1, fy1, fx2, fy2] = s.box;
                if (pos.x < fx1 * W - pad || pos.x > fx2 * W + pad) return;
                if (pos.y < fy1 * H - pad || pos.y > fy2 * H + pad) return;
                const cx = (fx1 + fx2) / 2 * W, cy = (fy1 + fy2) / 2 * H;
                const d = (pos.x - cx) ** 2 + (pos.y - cy) ** 2;
                if (d < bestDist) { best = i; bestDist = d; }
            });
            return best;
        }

        function playClip(b64) {
            if (!b64) return;
            voicePlayer.pause();
            audioQueue = [b64];
            isPlaying = false;
            playNextAudio();
        }

        function exploreAt(pos, force) {
            const idx = symbolAt(pos);
            const now = Date.now();
            if (idx === lastSymbolIndex) return;
            if (!force && now - lastExploreTime < 200) return;
            lastSymbolIndex = idx;
            lastExploreTime = now;

            if (idx >= 0) {
                const s = symbolMap.segments[idx];
                setStatus(s.label);
                playClip(s.audio_b64);
                return;
            }
            // Empty space: say which direction the nearest symbol is
            let clip = symbolMap.nothing, text = "Nothing here.";
            let nearest = null, nd = Infinity;
            symbolMap.segments.forEach(s => {
                const cx = (s.box[0] + s.box[2]) / 2 * canvas.width, cy = (s.box[1] + s.box[3]) / 2 * canvas.height;
                const d = (pos.x - cx) ** 2 + (pos.y - cy) ** 2;
                if (d < nd) { nd = d; nearest = cx; }
            });
            if (nearest !== null) {
                if (nearest < pos.x) { clip = symbolMap.left; text = "Nothing here. Try moving left."; }
                else { clip = symbolMap.right; text = "Nothing here. Try moving right."; }
            }
            setStatus(text);
            playClip(clip);
        }

        function requestMap() {
            const now = Date.now();
            if (now - lastMapRequest < 3000) return;   // a request is already in progress
            lastMapRequest = now;
            setStatus("Checking what you wrote...");
            ws.send(JSON.stringify({ action: "map_canvas", image_b64: getSnapshot(), version: strokeVersion, announce: true }));
        }

        function startExplore(e) {
            if (isDrawing || isRecordingHelp) return;
            if (Date.now() - lastPenTime < 700) return;   // ignore the palm right after the pencil lifts
            if (touchActiveId !== null) return;
            e.preventDefault();
            touchActiveId = e.pointerId;
            lastSymbolIndex = -2;
            try { canvas.setPointerCapture(e.pointerId); } catch (err) {}

            if (symbolMap.version !== strokeVersion) {
                requestMap();                              // drawing changed since the last map
                return;
            }
            exploreAt(getPos(e), true);
        }

        function moveExplore(e) {
            if (e.pointerId !== touchActiveId) return;
            e.preventDefault();
            if (symbolMap.version !== strokeVersion) return;
            exploreAt(getPos(e), false);
        }

        function endExplore(e) {
            if (e.pointerId !== touchActiveId) return;
            touchActiveId = null;
            lastSymbolIndex = -2;
        }

        // ---------- Drawing (pencil only; a finger explores instead) ----------
        let activePointerId = null;
        let activePointerType = "";

        function addPoint(pos) {
            // Stroke one segment at a time; re-stroking the whole path on every move is slow and smears
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);

            const pt = [Math.round(pos.x), Math.round(pos.y)];
            chunkPath.push(pt);
            currentStroke.push(pt);
        }

        function finishStroke() {
            if (!isDrawing) return;
            isDrawing = false;
            activePointerId = null;
            activePointerType = "";

            if (chunkPath.length > 0) {
                ws.send(JSON.stringify({ action: "draw_stream", path: chunkPath, canvas_width: canvas.width }));
                chunkPath = [];
            }

            if (currentStroke.length > 1) { strokeHistory.push([...currentStroke]); markCanvasChanged(); }
            lastPenTime = Date.now();

            // Ask for feedback after 4 seconds without a new stroke
            pendingStrokes = true;
            evaluationTimeout = setTimeout(() => {
                if (pendingStrokes && !isRecordingHelp) {
                    ws.send(JSON.stringify({ action: "evaluate", image_b64: getSnapshot(), version: strokeVersion }));
                    pendingStrokes = false;
                }
            }, 4000);
        }

        canvas.addEventListener("pointerdown", (e) => {
            if (mode !== "draw") return;
            if (e.pointerType === "touch") { startExplore(e); return; }   // finger explores, never draws
            if (!boardReady) return;                                       // no equation to follow yet

            if (isDrawing) {
                // The pencil takes priority over a palm or finger that started the stroke
                if (e.pointerType === "pen" && activePointerType !== "pen") finishStroke();
                else return;
            }

            clearTimeout(evaluationTimeout);
            isDrawing = true;
            activePointerId = e.pointerId;
            activePointerType = e.pointerType;
            // Keep receiving this pointer's events even if it leaves the canvas
            try { canvas.setPointerCapture(e.pointerId); } catch (err) {}

            chunkPath = [];
            currentStroke = [];
            const pos = getPos(e);
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
            const pt = [Math.round(pos.x), Math.round(pos.y)];
            chunkPath.push(pt);
            currentStroke.push(pt);
            e.preventDefault();
        });

        canvas.addEventListener("pointermove", (e) => {
            if (e.pointerType === "touch") { moveExplore(e); return; }
            if (e.pointerType === "pen" && e.buttons === 0) {
                const now = Date.now();
                if (now - lastHoverTime > 1500) {
                    ws.send(JSON.stringify({ action: "hover" }));
                    lastHoverTime = now;
                }
            }
            if (!isDrawing || e.pointerId !== activePointerId) return;

            // The Pencil samples ~240 Hz but the browser fires one event per frame.
            // Coalesced events recover the missing samples so quick strokes like "=" keep their shape.
            const samples = (e.getCoalescedEvents && e.getCoalescedEvents().length) ? e.getCoalescedEvents() : [e];
            for (const ev of samples) addPoint(getPos(ev));

            if (chunkPath.length >= 8) {
                ws.send(JSON.stringify({ action: "draw_stream", path: chunkPath, canvas_width: canvas.width }));
                chunkPath = [];
            }
            e.preventDefault();
        });

        canvas.addEventListener("pointerup", (e) => {
            if (e.pointerType === "touch") { endExplore(e); return; }
            if (!isDrawing || e.pointerId !== activePointerId) return;
            addPoint(getPos(e));   // a fast flick may produce no move events
            finishStroke();
        });

        // iPadOS can cancel a stroke (system gesture, palm, etc.); finish it so it isn't lost
        canvas.addEventListener("pointercancel", (e) => {
            if (e.pointerType === "touch") { endExplore(e); return; }
            if (e.pointerId === activePointerId) finishStroke();
        });

        canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    </script>
</body>
</html>
"""

@app.get("/")
async def get_ui():
    return HTMLResponse(html_template)

# --- 5. Phone camera page (open <server-url>/camera on the phone) ---
camera_html = """
<!DOCTYPE html>
<html>
<head>
    <title>FollowMath Camera</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        html, body { margin: 0; height: 100%; background: #000; color: #fff; font-family: sans-serif; overflow: hidden; }
        #cam-video { position: fixed; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; background: #000; }
        #info { position: fixed; left: 0; right: 0; bottom: 0; padding: 24px 20px 36px; background: rgba(0,0,0,0.7); font-size: 24px; font-weight: bold; text-align: center; }
        #dot { display: inline-block; width: 14px; height: 14px; border-radius: 50%; background: #f38ba8; margin-right: 8px; vertical-align: middle; }
        #dot.ok { background: #a6e3a1; }
        #start { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #11111b; color: #a6e3a1; z-index: 10; display: flex; align-items: center; justify-content: center; font-size: 28px; font-weight: bold; text-align: center; padding: 20px; box-sizing: border-box; touch-action: manipulation; }
    </style>
</head>
<body style="background:#000;color:#fff">
    <div id="start" style="background:#11111b;color:#a6e3a1">Tap anywhere to start the camera.<br><br>Then prop the phone up facing the whiteboard.</div>
    <video id="cam-video" autoplay playsinline muted></video>
    <div id="info"><span id="dot"></span><span id="state">Tap anywhere to start</span></div>
    <script>
        const video = document.getElementById("cam-video");
        const stateEl = document.getElementById("state");
        const dot = document.getElementById("dot");
        const startEl = document.getElementById("start");
        const LIVE_INTERVAL_MS = 8000;      // how often to send a frame while the iPad is in drawing mode
        const READY_TEXT = "Tap anywhere to take a picture";
        let ws = null, camStream = null, liveOn = false, liveTimer = null, wakeLock = null;
        let started = false, busy = false, busyTimer = null, audioCtx = null, scanned = false;

        function setState(t) { stateEl.innerText = t; }

        function beep(freq, dur) {
            try {
                if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (audioCtx.state === "suspended") audioCtx.resume();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.frequency.value = freq;
                gain.gain.value = 0.25;
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + dur);
            } catch (e) { console.error("Beep failed:", e); }
        }

        function grab(maxSide, quality) {
            const vw = video.videoWidth, vh = video.videoHeight;
            if (!vw || !vh) return null;
            const s = Math.min(1, maxSide / Math.max(vw, vh));
            const tmp = document.createElement("canvas");
            tmp.width = Math.round(vw * s);
            tmp.height = Math.round(vh * s);
            tmp.getContext("2d").drawImage(video, 0, 0, tmp.width, tmp.height);
            return tmp.toDataURL("image/jpeg", quality);
        }

        function setBusy(b) {
            busy = b;
            clearTimeout(busyTimer);
            if (b) busyTimer = setTimeout(() => { busy = false; setState(READY_TEXT); }, 45000);   // safety reset
        }

        // Three low beeps (hold still), then a high shutter beep and the capture
        function takePicture(fromIpad) {
            if (busy || !camStream) return;
            if (!fromIpad && (scanned || liveOn)) return;   // once the board is being watched, taps do nothing
            setBusy(true);
            setState("Hold still...");
            beep(500, 0.12);
            setTimeout(() => beep(500, 0.12), 600);
            setTimeout(() => beep(500, 0.12), 1200);
            setTimeout(() => {
                beep(1100, 0.25);   // shutter sound
                const img = grab(1600, 0.9);
                if (img && ws && ws.readyState === 1) {
                    setState("Scanning the whiteboard...");
                    ws.send(JSON.stringify({ action: "scan_image", image_b64: img }));
                } else {
                    setBusy(false);
                    setState("Camera not ready. Tap to try again.");
                }
            }, 1800);
        }

        function sendLiveFrame() {
            if (!liveOn || !ws || ws.readyState !== 1) return;
            const img = grab(1280, 0.85);
            if (img) ws.send(JSON.stringify({ action: "live_frame", image_b64: img }));
        }

        function setLive(on) {
            liveOn = on;
            if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
            if (on) liveTimer = setInterval(sendLiveFrame, LIVE_INTERVAL_MS);
            if (!busy) setState(on ? "Watching the board" : (scanned ? "Watching the board" : READY_TEXT));
        }

        function connect() {
            const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(proto + "//" + window.location.host + "/ws/camera");
            ws.onopen = () => { dot.classList.add("ok"); if (!busy) setState(READY_TEXT); };
            ws.onclose = () => { dot.classList.remove("ok"); setState("Reconnecting..."); setTimeout(connect, 2000); };
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.action === "capture") {
                    takePicture(true);                   // the iPad requested a new picture
                } else if (msg.action === "live") {
                    setLive(!!msg.on);
                } else if (msg.action === "info") {
                    setState(msg.text);
                } else if (msg.action === "scan_done") {
                    setBusy(false);
                    if (msg.ok) scanned = true;
                    setState(msg.text);
                }
            };
        }

        async function startCamera() {
            try {
                camStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
                    audio: false
                });
                video.srcObject = camStream;
                await video.play();
                setState(READY_TEXT);
            } catch (err) {
                console.error("Camera error:", err);
                setState("Camera error: " + err);
                const send = () => ws.send(JSON.stringify({ action: "camera_error" }));
                if (ws && ws.readyState === 1) send(); else if (ws) ws.addEventListener("open", send, { once: true });
            }
            try { wakeLock = await navigator.wakeLock.request("screen"); } catch (e) {}   // keep the screen awake
        }

        // The first tap starts the camera and unlocks audio for the beeps
        startEl.addEventListener("pointerdown", (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (started) return;
            started = true;
            startEl.style.display = "none";
            try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); audioCtx.resume(); } catch (er) {}
            setState("Starting the camera...");
            connect();
            startCamera();
            setInterval(() => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ action: "ping" })); }, 20000);   // keepalive
        });

        document.addEventListener("pointerdown", (e) => {
            if (!started) return;
            e.preventDefault();
            takePicture();
        });
        document.addEventListener("visibilitychange", async () => {
            // Wake locks are released when the page is hidden, so request it again
            if (document.visibilityState === "visible" && camStream) {
                try { wakeLock = await navigator.wakeLock.request("screen"); } catch (e) {}
            }
        });
    </script>
</body>
</html>
"""

@app.get("/camera")
async def get_camera_ui():
    return HTMLResponse(camera_html)

@app.websocket("/ws/camera")
async def camera_endpoint(websocket: WebSocket):
    # Phone side: only captures images. Frames are forwarded to the iPad session, which does the processing.
    await websocket.accept()
    hub["phone_ws"] = websocket
    print("Phone camera connected.")
    await push_phone_live()
    audio = await get_fast_voice(PHONE_ON)
    await notify_ipad({"action": "phone_status", "connected": True, "message": PHONE_ON,
                       "audio_b64": audio, "priority": False})
    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            if payload.get("action") in ("scan_image", "live_frame", "camera_error"):
                inbox = hub["inbox"]
                if inbox is not None:
                    inbox.put_nowait(payload)
                elif payload.get("action") == "scan_image":
                    await websocket.send_json({"action": "scan_done", "ok": False,
                                               "text": "The iPad is not connected. Open the iPad page first."})
                else:
                    print(f"   Phone sent {payload.get('action')} but no iPad page is connected, dropped")
    except WebSocketDisconnect:
        print("Phone camera disconnected.")
    except Exception as e:
        print(f"Phone camera connection closed: {e}")
    finally:
        if hub["phone_ws"] is websocket:
            hub["phone_ws"] = None
            audio = await get_fast_voice(PHONE_OFF)
            await notify_ipad({"action": "phone_status", "connected": False, "message": PHONE_OFF,
                               "audio_b64": audio, "priority": False})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # The iPad's own messages and the phone's camera frames share one queue,
    # so the session loop below handles both the same way.
    inbox = asyncio.Queue()
    hub.setdefault("sessions", []).append((websocket, inbox))
    hub["ipad_ws"] = websocket
    hub["inbox"] = inbox
    hub["live_on"] = False
    print(f"iPad session opened ({len(hub['sessions'])} open)")

    async def pump():
        # Forward incoming iPad messages into the queue
        try:
            while True:
                inbox.put_nowait(json.loads(await websocket.receive_text()))
        except Exception:
            pass
        finally:
            inbox.put_nowait(None)      # signals the main loop that the iPad disconnected

    pump_task = asyncio.create_task(pump())
    await websocket.send_json({"action": "phone_status", "connected": hub["phone_ws"] is not None})

    anchor_x = None
    anchor_y = None
    scale_factor = 1.0
    active_stencil = None          # set once an equation has been scanned

    last_spoken_direction = ""
    last_audio_time = 0

    # --- Touchable symbol map: SAM segments the drawing and Spark names each symbol ---
    map_ctx = {"task": None, "version": None, "announce": False}
    live_task = None               # at most one live board read in flight
    drawn_points = []              # every point drawn, kept so strokes can be replayed if the board changes
    canvas_width_seen = 800

    async def run_map(img_b64, version):
        try:
            segs = await build_symbol_map(img_b64, pipeline_data.get("semantic_text", ""))
            audios = await asyncio.gather(*(get_fast_voice(s["label"]) for s in segs))
            for s, a in zip(segs, audios):
                s["audio_b64"] = a
            nothing, left, right = await asyncio.gather(
                get_fast_voice(NOTHING_HERE), get_fast_voice(NOTHING_LEFT), get_fast_voice(NOTHING_RIGHT))
            announce = map_ctx["announce"]
            await websocket.send_json({"action": "symbol_map", "version": version, "segments": segs,
                                       "nothing_audio": nothing, "left_audio": left, "right_audio": right})
            if announce:
                n = len(segs)
                if n == 0:
                    summary = MAP_EMPTY
                else:
                    summary = (f"I found {n} symbol{'s' if n != 1 else ''}. "
                               "Slide one finger over what you wrote to hear each one.")
                audio = await get_fast_voice(summary)
                await websocket.send_json({"message": summary, "audio_b64": audio, "priority": True})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Symbol map failed: {e}")
            try:
                await websocket.send_json({"action": "symbol_map_failed"})
            except Exception:
                pass

    def start_map(img_b64, version, announce):
        t = map_ctx["task"]
        if t and not t.done() and map_ctx["version"] == version:
            map_ctx["announce"] = map_ctx["announce"] or announce   # this version is already being mapped
            return
        if t and not t.done():
            t.cancel()
        map_ctx.update(version=version, announce=announce)
        map_ctx["task"] = asyncio.create_task(run_map(img_b64, version))

    async def run_live(img_b64):
        # Re-read the board after the professor writes more. If the equation changed, rebuild the stencil
        # (keeping the student's drawing) and announce what is new. Stays silent if nothing changed.
        nonlocal active_stencil, scale_factor, furthest_x, next_evaluation_checkpoint, last_spoken_direction
        try:
            old_text = pipeline_data["semantic_text"]
            result = await process_scan(img_b64, live=True)
            if not result.get("changed"):
                return
            new_text = pipeline_data["semantic_text"]
            print(f"   Live frame: board changed, announcing: {new_text}")

            # Rebuild the stencil for the longer equation and replay the student's strokes onto it
            active_stencil = pipeline_data["raw_stencil_mask"].copy()
            furthest_x = pipeline_data["start_x"]
            stencil_width = pipeline_data["x2"] - pipeline_data["x1"]
            if anchor_x is not None:
                if stencil_width > 0:
                    scale_factor = max(0.15, (canvas_width_seen - anchor_x - 50) / stencil_width)
                for px, py in drawn_points:
                    nx = int((px - anchor_x) / scale_factor) + pipeline_data["start_x"]
                    ny = int((py - anchor_y) / scale_factor) + pipeline_data["start_y"]
                    cv2.circle(active_stencil, (nx, ny), 20, 0, -1)
                    if nx > furthest_x:
                        furthest_x = nx
            if stencil_width > 0:
                progress = max(0.0, (furthest_x - pipeline_data["start_x"]) / stencil_width)
                next_evaluation_checkpoint = (int(progress / 0.25) + 1) * 0.25

            # If the old text is a prefix of the new text, only announce the addition
            if not old_text:
                msg = f"The professor started writing. It reads: {new_text}."
            elif new_text.lower().startswith(old_text.lower()) and len(new_text) > len(old_text):
                added = new_text[len(old_text):].strip(" ,.")
                msg = f"The board was updated. New: {added}." if added else f"The board was updated. It now reads: {new_text}."
            else:
                msg = f"The board was updated. It now reads: {new_text}."
            audio = await get_fast_voice(msg)
            await websocket.send_json({"action": "live_update", "message": msg, "audio_b64": audio, "priority": True,
                                       "crop_b64": result.get("crop_b64", ""), "semantic_text": new_text})
            last_spoken_direction = msg
        except asyncio.CancelledError:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Live update failed: {e}")

    furthest_x = 0
    next_evaluation_checkpoint = 0.25   # tutor feedback at 25%, 50%, 75% and 100% progress
    eval_task = None                    # at most one canvas evaluation in flight

    try:
        while True:
            payload = await inbox.get()
            if payload is None:
                raise WebSocketDisconnect()
            action = payload.get("action")
            now = time.time()
            if action not in ("draw_stream", "hover"):
                print(f"WS action received: {action}")

            # ---------- Scanning ----------
            if action == "request_intro":
                audio = await get_fast_voice(SCAN_INTRO)
                await websocket.send_json({"message": SCAN_INTRO, "audio_b64": audio, "priority": True})
                last_spoken_direction = SCAN_INTRO
                continue

            if action == "camera_error":
                msg = CAMERA_ERROR_MSG
                audio = await get_fast_voice(msg)
                await websocket.send_json({"message": msg, "audio_b64": audio, "priority": True})
                continue

            if action == "live_control":
                hub["live_on"] = bool(payload.get("on"))
                await push_phone_live()
                continue

            if action == "scan_timeout":
                audio = await get_fast_voice(PHONE_TIMEOUT)
                await websocket.send_json({"message": PHONE_TIMEOUT, "audio_b64": audio, "priority": True})
                continue

            if action == "scan_request":
                # The student asked for a new scan: tell the phone to take the picture
                phone = hub["phone_ws"]
                sent = False
                if phone is not None:
                    try:
                        await phone.send_json({"action": "capture"})
                        sent = True
                    except Exception:
                        pass
                if not sent:
                    audio = await get_fast_voice(PHONE_MISSING)
                    await websocket.send_json({"action": "scan_result", "ok": False, "message": PHONE_MISSING,
                                               "audio_b64": audio, "priority": True})
                continue

            if action == "scan_image":
                image_data = payload.get("image_b64")
                if not image_data:
                    continue
                if eval_task and not eval_task.done():
                    eval_task.cancel()

                if map_ctx["task"] and not map_ctx["task"].done():
                    map_ctx["task"].cancel()
                if live_task and not live_task.done():
                    live_task.cancel()
                audio = await get_fast_voice(SCAN_WAIT)
                await websocket.send_json({"message": SCAN_WAIT, "audio_b64": audio, "priority": True})
                await notify_phone({"action": "info", "text": "Reading the equation..."})

                result = await process_scan(image_data)
                if not result["ok"]:
                    phone_text = "Could not read it. Tap anywhere to try again."
                elif result.get("waiting"):
                    phone_text = "Board position saved. Watching the board."
                else:
                    phone_text = "Equation read. Watching the board."
                await notify_phone({"action": "scan_done", "ok": result["ok"], "text": phone_text})

                if result["ok"]:
                    # Reset session state for the new equation
                    drawn_points.clear()
                    active_stencil = pipeline_data["raw_stencil_mask"].copy()
                    anchor_x = None
                    anchor_y = None
                    scale_factor = 1.0
                    furthest_x = pipeline_data["start_x"]
                    next_evaluation_checkpoint = 0.25
                    last_spoken_direction = ""

                audio = await get_fast_voice(result["message"])
                await websocket.send_json({
                    "action": "scan_result",
                    "ok": result["ok"],
                    "message": result["message"],
                    "audio_b64": audio,
                    "priority": True,
                    "crop_b64": result.get("crop_b64", ""),
                    "semantic_text": result.get("semantic_text", ""),
                    "waiting": result.get("waiting", False),
                })
                continue

            # Everything below requires a scanned equation
            if active_stencil is None:
                continue

            if action == "clear_canvas":
                anchor_x = None
                anchor_y = None
                drawn_points.clear()
                active_stencil = pipeline_data["raw_stencil_mask"].copy()
                furthest_x = pipeline_data["start_x"]
                next_evaluation_checkpoint = 0.25

                audio = await get_fast_voice("Canvas cleared. Let's start over.")
                await websocket.send_json({"message": "Canvas cleared.", "audio_b64": audio, "priority": True})
                continue

            if action == "undo_canvas":
                if payload.get("nothing_to_undo"):
                    msg = "There is nothing to undo."
                    audio = await get_fast_voice(msg)
                    await websocket.send_json({"message": msg, "audio_b64": audio, "priority": True})
                    continue
                history = payload.get("history", [])
                drawn_points.clear()
                active_stencil = pipeline_data["raw_stencil_mask"].copy()
                furthest_x = pipeline_data["start_x"]

                # Rebuild the stencil from the strokes that remain
                for stroke in history:
                    for px, py in stroke:
                        drawn_points.append((px, py))
                        if anchor_x is not None:
                            norm_x = int((px - anchor_x) / scale_factor) + pipeline_data["start_x"]
                            norm_y = int((py - anchor_y) / scale_factor) + pipeline_data["start_y"]
                            cv2.circle(active_stencil, (norm_x, norm_y), 20, 0, -1)
                            if norm_x > furthest_x:
                                furthest_x = norm_x

                audio = await get_fast_voice("Undone.")
                await websocket.send_json({"message": "Undone.", "audio_b64": audio, "priority": True})
                continue

            if action == "live_frame":
                frame = payload.get("image_b64")
                if frame and live_task is not None and not live_task.done():
                    print("   Live frame dropped: previous board read still running")
                if frame and (live_task is None or live_task.done()):
                    live_task = asyncio.create_task(run_live(frame))
                continue

            if action == "map_canvas":
                # The student touched their writing and the symbol map is out of date
                map_img = payload.get("image_b64")
                if map_img:
                    audio = await get_fast_voice(MAP_WAIT)
                    await websocket.send_json({"message": MAP_WAIT, "audio_b64": audio, "priority": True})
                    start_map(map_img, payload.get("version", 0), True)
                continue

            if action == "evaluate":
                if not pipeline_data["ready"]:
                    continue
                image_data = payload.get("image_b64")
                # Refresh the symbol map from the same image (runs in the background)
                if image_data:
                    start_map(image_data, payload.get("version", 0), False)
                # One evaluation at a time; extra requests are dropped, not queued
                if image_data and (eval_task is None or eval_task.done()):
                    async def run_eval(img):
                        nonlocal last_spoken_direction
                        msg = "Let me check what you have written so far..."
                        audio = await get_fast_voice(msg)
                        await websocket.send_json({"message": msg, "audio_b64": audio, "priority": True})

                        tutor_feedback = await evaluate_canvas_with_spark(img, pipeline_data["semantic_text"])
                        audio = await get_fast_voice(tutor_feedback)
                        await websocket.send_json({"message": tutor_feedback, "audio_b64": audio, "priority": True})
                        last_spoken_direction = tutor_feedback

                    eval_task = asyncio.create_task(run_eval(image_data))
                continue

            if action == "ask_tutor":
                # A spoken question takes priority over any running evaluation
                if eval_task and not eval_task.done():
                    eval_task.cancel()

                audio_data = payload.get("audio_b64")
                image_data = payload.get("image_b64")

                if audio_data and image_data:
                    # 1. Transcribe the question
                    user_question = await transcribe_audio(audio_data)
                    user_lower = user_question.lower()

                    if not user_question:
                        err = await get_fast_voice("I didn't quite catch that.")
                        await websocket.send_json({"message": "Audio failed.", "audio_b64": err, "priority": True})
                        continue

                    # Voice commands
                    if any(k in user_lower for k in ("new equation", "another equation", "scan", "retake",
                                                     "take a picture", "take a photo", "new picture", "new photo")):
                        await websocket.send_json({"action": "force_rescan"})
                        continue
                    if "undo" in user_lower:
                        await websocket.send_json({"action": "force_undo"})
                        continue
                    elif "erase" in user_lower or "clear" in user_lower:
                        await websocket.send_json({"action": "force_clear"})
                        continue

                    # 2. Otherwise, answer the question using the current drawing as context
                    ack_audio = await get_fast_voice("Thinking...")
                    await websocket.send_json({"message": "Thinking...", "audio_b64": ack_audio, "priority": True})

                    headers = {"Authorization": f"Bearer {META_API_KEY}", "Content-Type": "application/json"}
                    spark_payload = {
                        "model": "muse-spark-1.3",
                        "messages": [
                            {
                                "role": "system",
                                "content": f"You are a math tutor. The target equation is '{pipeline_data['semantic_text']}'. Look at the user's drawing and answer their question clearly and concisely in 1-2 sentences for text-to-speech."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"My question is: {user_question}"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data.split(',')[1]}"}}
                                ]
                            }
                        ]
                    }

                    async with httpx.AsyncClient() as client:
                        try:
                            response = await client.post(META_CHAT_ENDPOINT, headers=headers, json=spark_payload, timeout=15.0)
                            if response.status_code == 200:
                                answer = response.json()["choices"][0]["message"]["content"].strip()
                                answer_audio = await get_fast_voice(answer)
                                await websocket.send_json({"message": answer, "audio_b64": answer_audio, "priority": True})
                                last_spoken_direction = answer
                            else:
                                print(f"Spark Q&A API error: {response.status_code} - {response.text}")
                                fail_audio = await get_fast_voice("Sorry, I couldn't answer that. Please try again.")
                                await websocket.send_json({"message": "Spark Q&A failed.", "audio_b64": fail_audio, "priority": True})
                        except Exception as e:
                            print(f"Spark Q&A failed: {e}")
                            fail_audio = await get_fast_voice("Sorry, something went wrong. Please try again.")
                            await websocket.send_json({"message": "Spark Q&A failed.", "audio_b64": fail_audio, "priority": True})
                continue

            if action == "hover":
                if not pipeline_data["ready"]:
                    continue
                if anchor_x is None:
                    msg = "Ready. Tap anywhere to anchor."
                    if msg != last_spoken_direction:
                        audio = await get_fast_voice(msg)
                        await websocket.send_json({"message": msg, "audio_b64": audio, "priority": True})
                        last_spoken_direction = msg
                continue

            path = payload.get("path", [])
            canvas_width_seen = payload.get("canvas_width", canvas_width_seen)
            if not pipeline_data["ready"]:
                # No equation on the board yet, so there is nothing to follow
                if now - last_audio_time > 10.0:
                    audio = await get_fast_voice(NOT_READY_MSG)
                    await websocket.send_json({"message": NOT_READY_MSG, "audio_b64": audio, "priority": True})
                    last_audio_time = now
                continue
            if not path:
                continue

            if anchor_x is None:
                # First stroke: anchor the stencil to where the student started and scale it to fit the canvas
                anchor_x, anchor_y = path[0]
                canvas_width = payload.get("canvas_width", 800)
                available_width = canvas_width - anchor_x
                stencil_width = pipeline_data["x2"] - pipeline_data["x1"]

                if stencil_width > 0:
                    scale_factor = max(0.15, (available_width - 50) / stencil_width)

                msg = "Equation anchored! Start writing."
                audio = await get_fast_voice(msg)
                await websocket.send_json({"message": msg, "audio_b64": audio, "priority": True})
                last_spoken_direction = msg
                continue

            for px, py in path:
                drawn_points.append((px, py))
                norm_x = int((px - anchor_x) / scale_factor) + pipeline_data["start_x"]
                norm_y = int((py - anchor_y) / scale_factor) + pipeline_data["start_y"]
                cv2.circle(active_stencil, (norm_x, norm_y), 20, 0, -1)   # erase the covered part of the stencil

                # Track the furthest horizontal progress
                if norm_x > furthest_x:
                    furthest_x = norm_x

            # Request tutor feedback each time progress crosses a 25% checkpoint
            stencil_width = pipeline_data["x2"] - pipeline_data["x1"]
            if stencil_width > 0:
                current_progress = (furthest_x - pipeline_data["start_x"]) / stencil_width
                if current_progress >= next_evaluation_checkpoint:
                    await websocket.send_json({"action": "request_snapshot"})
                    # Skip ahead to the next checkpoint past the current progress
                    next_evaluation_checkpoint = (int(current_progress / 0.25) + 1) * 0.25

            last_px, last_py = path[-1]
            norm_last_x = int((last_px - anchor_x) / scale_factor) + pipeline_data["start_x"]
            norm_last_y = int((last_py - anchor_y) / scale_factor) + pipeline_data["start_y"]

            nav_text = get_smart_navigation(norm_last_x, norm_last_y, active_stencil)

            # Speak only to correct drift; repeat a correction at most every 8 seconds
            if nav_text and (nav_text != last_spoken_direction or
                             (nav_text.startswith(("You drifted", "Move right")) and now - last_audio_time > 8.0)):
                audio = await get_fast_voice(nav_text)
                await websocket.send_json({"message": f"{nav_text}", "audio_b64": audio, "priority": False})
                last_spoken_direction = nav_text
                last_audio_time = now

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Connection closed due to error: {e}")
    finally:
        pump_task.cancel()
        hub["sessions"] = [x for x in hub.get("sessions", []) if x[1] is not inbox]
        if hub["inbox"] is inbox:
            if hub["sessions"]:
                # Another iPad page is still open: hand the phone's frames to it
                hub["ipad_ws"], hub["inbox"] = hub["sessions"][-1]
            else:
                hub["ipad_ws"] = None
                hub["inbox"] = None
            hub["live_on"] = False
            await push_phone_live()
        print(f"iPad session closed ({len(hub['sessions'])} open)")
        if map_ctx["task"] and not map_ctx["task"].done():
            map_ctx["task"].cancel()
        if live_task and not live_task.done():
            live_task.cancel()
        if eval_task and not eval_task.done():
            eval_task.cancel()
