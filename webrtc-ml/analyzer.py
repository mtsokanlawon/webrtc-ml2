# analyzer.py
# import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    BeginEvent,
    TurnEvent,
    TerminationEvent,
    StreamingError,
)

import base64
import threading
import subprocess
from queue import Queue, Empty

import os, io, re
import cv2
import time, math, base64, shutil, subprocess
import asyncio, websockets, json
print(websockets.__version__)
print(websockets.__file__)
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Tuple

import numpy as np
from flask import Flask, Blueprint, request, jsonify, send_file
import io
import soundfile as sf
from faster_whisper import WhisperModel
import mediapipe as mp

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

# Manage ongoing sessions globally
sessions = {}

# A global or per-room store (could be replaced with a DB)
engagement_events = defaultdict(list)  # { roomId: [ {peerId, eventType, description, timestamp} ] }
transcript_logs = defaultdict(list)     # { roomId: [ {text, timestamp} ] }
# peer_map = defaultdict(lambda: None)

analyzer_bp = Blueprint("analyzer", __name__)
app = Flask(__name__)

ASSEMBLYAI_API_KEY = "433a0641544a46cfab94eb5b44ec4f1f"

DATA_ROOT = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_ROOT, exist_ok=True)

AUDIO_SAMPLE_RATE = 16000

# ===== Engagement thresholds =====
VIDEO_FPS_TARGET = 30
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES_CLOSED = 3
EAR_CONSEC_FRAMES_OPEN = 2
MAR_THRESHOLD = 0.7
MAR_CONSEC_FRAMES_OPEN = 5

ATTENTION_YAW_THRESHOLD = 25
PITCH_FOCUSED_MIN_ABS_THRESHOLD = 90
ATTENTION_CONSISTENCY_SECONDS = 3

HAND_RAISE_Y_THRESHOLD_FACTOR = 0.4
HAND_MOVEMENT_WINDOW_FRAMES = 90
HAND_MOVEMENT_STD_THRESHOLD = 0.04
HAND_MOVEMENT_COOLDOWN_SECONDS = 15
HAND_RAISE_COOLDOWN_SECONDS = 5

FATIGUE_YAWN_COUNT = 2
FATIGUE_YAWN_WINDOW_SECONDS = 60
FATIGUE_YAWN_COOLDOWN_SECONDS = 60

FATIGUE_BLINK_COUNT = 5
FATIGUE_BLINK_WINDOW_SECONDS = 10
FATIGUE_BLINK_COOLDOWN_SECONDS = 10

# ===== Utils =====
def room_peer_dir(room_id, peer_id):
    d = os.path.join(DATA_ROOT, room_id, peer_id)
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    os.makedirs(os.path.join(d, "audio"), exist_ok=True)
    os.makedirs(os.path.join(d, "reports"), exist_ok=True)
    return d

def ts_now():
    return datetime.now().strftime("%H:%M:%S")

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_eye_aspect_ratio(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3]) or 1e-9
    return (A + B) / (2.0 * C)

def get_mouth_aspect_ratio(mouth):
    A = euclidean_distance(mouth[1], mouth[5])
    B = euclidean_distance(mouth[2], mouth[4])
    C = euclidean_distance(mouth[0], mouth[3]) or 1e-9
    return (A + B) / (2.0 * C)

def head_pose(face_landmarks, image_shape):
    img_h, img_w = image_shape[:2]
    pts = np.array([
        (face_landmarks[1].x * img_w,   face_landmarks[1].y * img_h),   # Nose tip
        (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h), # Chin
        (face_landmarks[33].x * img_w,  face_landmarks[33].y * img_h),  # Left eye corner
        (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h), # Right eye corner
        (face_landmarks[61].x * img_w,  face_landmarks[61].y * img_h),  # Left mouth corner
        (face_landmarks[291].x * img_w, face_landmarks[291].y * img_h), # Right mouth corner
    ], dtype="double")

    model = np.array([
        (0.0,   0.0,   0.0),
        (0.0,  -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0,  170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ])
    focal = img_w
    center = (img_w/2, img_h/2)
    K = np.array([[focal,0,center[0]],
                  [0,focal,center[1]],
                  [0,0,1]], dtype="double")
    dist = np.zeros((4,1))
    ok, rvec, tvec = cv2.solvePnP(model, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(R)
    pitch, yaw, roll = angles  # already in degrees
    return pitch, yaw, roll

# ===== Engagement state machine (batch mode) =====
class EngagementLogic:
    def __init__(self, logger):
        self.logger = logger
        self._state = "Focused"
        self._distraction_start = 0.0
        self.last_hand_raised = 0.0
        self.hand_cooldown_end = 0.0

        self.yawn_q = deque()
        self.yawn_cd_end = 0.0
        self.blink_q = deque()
        self.blink_cd_end = 0.0

        self._is_eye_closed = False
        self._frames_eye_closed = 0
        self._is_mouth_open = False
        self._frames_mouth_open = 0

    def _now(self, t):
        return t

    def update_attention(self, is_focused, yaw, pitch, t):
        now = self._now(t)
        if self._state == "Focused":
            if not is_focused:
                self._distraction_start = now
                self._state = "Distracted"
        elif self._state == "Distracted":
            if is_focused:
                self._state = "Focused"
            else:
                if (now - self._distraction_start) >= ATTENTION_CONSISTENCY_SECONDS:
                    direction = ""
                    if abs(yaw) > ATTENTION_YAW_THRESHOLD:
                        direction = "sideways"
                    if abs(pitch) < PITCH_FOCUSED_MIN_ABS_THRESHOLD:
                        direction = ("down" if pitch > 0 else "up") if not direction else direction + (" and down" if pitch > 0 else " and up")
                    self.logger(now, "Attention", f"Distracted{(' (looking ' + direction + ')') if direction else ''}")
                    self._state = "Logged_Distraction"
        elif self._state == "Logged_Distraction":
            if is_focused:
                self.logger(now, "Attention", "Focused")
                self._state = "Focused"

    def register_blink(self, ear, t):
        now = self._now(t)
        if now < self.blink_cd_end: return
        if ear < EAR_THRESHOLD:
            self._frames_eye_closed += 1
            if not self._is_eye_closed and self._frames_eye_closed >= EAR_CONSEC_FRAMES_CLOSED:
                self._is_eye_closed = True
        else:
            if self._is_eye_closed:
                if self._frames_eye_closed >= EAR_CONSEC_FRAMES_CLOSED:
                    self.blink_q.append(now)
                    while self.blink_q and now - self.blink_q[0] > FATIGUE_BLINK_WINDOW_SECONDS:
                        self.blink_q.popleft()
                    if len(self.blink_q) >= FATIGUE_BLINK_COUNT:
                        self.logger(now, "Fatigue", "Blink: tired fatigue")
                        self.blink_cd_end = now + FATIGUE_BLINK_COOLDOWN_SECONDS
                self._is_eye_closed = False
            self._frames_eye_closed = 0

    def register_yawn(self, mar, t):
        now = self._now(t)
        if now < self.yawn_cd_end: return
        if mar > MAR_THRESHOLD:
            self._frames_mouth_open += 1
            if not self._is_mouth_open and self._frames_mouth_open >= MAR_CONSEC_FRAMES_OPEN:
                self._is_mouth_open = True
        else:
            if self._is_mouth_open:
                if self._frames_mouth_open >= MAR_CONSEC_FRAMES_OPEN:
                    self.yawn_q.append(now)
                    while self.yawn_q and now - self.yawn_q[0] > FATIGUE_YAWN_WINDOW_SECONDS:
                        self.yawn_q.popleft()
                    if len(self.yawn_q) >= FATIGUE_YAWN_COUNT:
                        self.logger(now, "Fatigue", "Yawning")
                        self.yawn_cd_end = now + FATIGUE_YAWN_COOLDOWN_SECONDS
                self._is_mouth_open = False
            self._frames_mouth_open = 0

    def register_hand(self, is_raised, hand_std, t):
        now = self._now(t)
        if is_raised and (now - self.last_hand_raised) > HAND_RAISE_COOLDOWN_SECONDS:
            self.logger(now, "Hand Motion", "Hand Raised")
            self.last_hand_raised = now
            self.hand_cooldown_end = now + HAND_MOVEMENT_COOLDOWN_SECONDS
            return
        if now < self.hand_cooldown_end:
            return
        if hand_std is not None and hand_std > HAND_MOVEMENT_STD_THRESHOLD:
            self.logger(now, "Hand Motion", "Hand Detected")
            self.hand_cooldown_end = now + HAND_MOVEMENT_COOLDOWN_SECONDS


# ========= Plotting Helpers =========

def plot_stacked_timeline(engagements):
    participants = sorted(set(e.get("peerId", "Unknown") for e in engagements))
    color_map = {"Fatigue": "red", "Distracted": "orange", "HandRaise": "green"}
    symbol_map = {"Fatigue": "o", "Distracted": "s", "HandRaise": "^"}

    plt.figure(figsize=(7, 3))
    for i, peer in enumerate(participants):
        peer_events = [e for e in engagements if e.get("peerId") == peer]
        times = list(range(len(peer_events)))
        types = [e["eventType"] for e in peer_events]
        for t, etype in zip(times, types):
            plt.scatter(t, [i], c=color_map.get(etype, "blue"),
                        marker=symbol_map.get(etype, "x"))
    plt.yticks(range(len(participants)), participants)
    plt.xlabel("Event Index (time-ordered)")
    plt.title("Stacked Engagement Timeline (per participant)")
    plt.tight_layout()
    return plt


def plot_event_frequency_timeline(engagements):
    color_map = {"Fatigue": "red", "Distracted": "orange", "HandRaise": "green"}
    grouped = defaultdict(lambda: defaultdict(int))

    for e in engagements:
        grouped[e["timestamp"]][e["eventType"]] += 1

    times = sorted(grouped.keys())
    if not times:
        return None

    plt.figure(figsize=(7, 3))
    for etype in ["Fatigue", "Distracted", "HandRaise"]:
        counts = [grouped[t][etype] for t in times]
        plt.plot(times, counts, label=etype, color=color_map[etype])
    plt.xticks(rotation=45, ha="right")
    plt.title("Event Frequency Over Time")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    return plt


def plot_event_proportions(engagements):
    by_type = defaultdict(int)
    for e in engagements:
        by_type[e["eventType"]] += 1

    if not by_type:
        return None

    labels = list(by_type.keys())
    sizes = list(by_type.values())

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%",
            colors=["red", "orange", "green"])
    plt.title("Event Proportions")
    plt.tight_layout()
    return plt


def plot_engagement_heatmap(engagements):
    participants = sorted(set(e.get("peerId", "Unknown") for e in engagements))
    times = sorted(set(e["timestamp"] for e in engagements))

    if not participants or not times:
        return None

    matrix = np.zeros((len(participants), len(times)))
    for e in engagements:
        i = participants.index(e.get("peerId", "Unknown"))
        j = times.index(e["timestamp"])
        matrix[i, j] += 1

    plt.figure(figsize=(8, 4))
    sns.heatmap(matrix, cmap="YlOrRd", xticklabels=times, yticklabels=participants)
    plt.title("Engagement Heatmap (Time vs Participants)")
    plt.xlabel("Time")
    plt.ylabel("Participants")
    plt.tight_layout()
    return plt


def plot_event_distribution(engagements):
    counts = defaultdict(int)
    for e in engagements:
        counts[e.get("peerId", "Unknown")] += 1

    if not counts:
        return None

    plt.figure(figsize=(5, 3))
    sns.boxplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title("Distribution of Events per Participant")
    plt.xlabel("Participant")
    plt.ylabel("Event Count")
    plt.tight_layout()
    return plt


# ========= Insights =========

def compute_insights(engagements):
    total_hand_raises = sum(1 for e in engagements if e["eventType"] == "HandRaise")
    total_fatigue = sum(1 for e in engagements if e["eventType"] == "Fatigue")

    times = sorted(set(e["timestamp"] for e in engagements))
    fatigue_rate = (total_fatigue / max(len(times), 1)) * 100

    by_time = defaultdict(int)
    for e in engagements:
        if e["eventType"] == "Distracted":
            by_time[e["timestamp"]] += 1
    peak_distraction = max(by_time, key=by_time.get, default="N/A")

    return {
        "Total Hand Raises": total_hand_raises,
        "% Time Fatigued": f"{fatigue_rate:.1f}%",
        "Peak Distraction Window": peak_distraction
    }


# ========= PDF Generator =========

def generate_meeting_pdf(room_id, transcripts, engagements, summary):
    pdf_dir = "meeting_reports"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{room_id}_report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=40)

    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    story.append(Paragraph(f"<b>Meeting Summary for {room_id}</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))

    # --- Overall Summary ---
    story.append(Paragraph("<b>Overall Summary:</b>", styles["Heading2"]))
    if summary:
        for line in summary.split("\n"):
            story.append(Paragraph(line.strip(), styles["Normal"]))
    else:
        story.append(Paragraph("No summary available.", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # --- Insights Cards ---
    insights = compute_insights(engagements)
    story.append(Paragraph("<b>Aggregate Insights:</b>", styles["Heading2"]))
    for key, value in insights.items():
        story.append(Paragraph(f"■ {key}: {value}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # --- Engagement Visualizations ---
    plots = [
        ("Stacked Timeline", plot_stacked_timeline(engagements)),
        ("Event Frequency Timeline", plot_event_frequency_timeline(engagements)),
        ("Event Proportions", plot_event_proportions(engagements)),
        ("Engagement Heatmap", plot_engagement_heatmap(engagements)),
        ("Event Distribution", plot_event_distribution(engagements)),
    ]

    for title, fig in plots:
        if fig:
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
            story.append(Image(buf, width=400, height=200))
            story.append(Spacer(1, 0.2 * inch))
            plt.close()

    # --- Engagement Events ---
    story.append(Paragraph("<b>Engagement Events:</b>", styles["Heading2"]))
    if engagements:
        for e in engagements:
            line = f"{e['timestamp']} - {e.get('peerId', 'Unknown')}: {e['eventType']} ({e['description']})"
            story.append(Paragraph(line, styles["Normal"]))
    else:
        story.append(Paragraph("No engagement events recorded.", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # --- Transcript ---
    story.append(Paragraph("<b>Transcript:</b>", styles["Heading2"]))
    if transcripts:
        for t in transcripts:
            line = f"{t.get('timestamp', '')} ({t.get('speaker', '')}): {t.get('text', '')}"
            story.append(Paragraph(line, styles["Normal"]))
    else:
        story.append(Paragraph("No transcript recorded.", styles["Normal"]))

    doc.build(story)
    return pdf_path

def generate_meeting_summary(transcripts, engagements):
    """
    Merge transcripts + engagement events into a readable meeting summary.
    """
    lines = []

    # Transcript summary (no timestamps here)
    if transcripts:
        lines.append("Meeting Transcript Summary:")
        for t in transcripts[:10]:  # first 10
            lines.append(f"{t['text']}")
        if len(transcripts) > 10:
            lines.append("... (more)\n")

    # Engagement insights
    if engagements:
        lines.append("Engagement Summary:")
        by_type = defaultdict(int)
        for e in engagements:
            by_type[e["eventType"]] += 1
        for eventType, count in by_type.items():
            lines.append(f"{eventType}: {count} times detected")

    return "\n".join(lines)


@app.route("/finalize/<room_id>", methods=["POST"])
def finalize_room(room_id):
    engagements = engagement_events.get(room_id, [])
    transcripts = transcript_logs.get(room_id, [])

    if not engagements and not transcripts:
        return jsonify({"error": "No data recorded"}), 404

    summary = generate_meeting_summary(transcripts, engagements)

    # Generate PDF
    pdf_path = generate_meeting_pdf(room_id, transcripts, engagements, summary)

    # Cleanup memory after finalizing
    engagement_events.pop(room_id, None)
    transcript_logs.pop(room_id, None)

    return jsonify({
        "room": room_id,
        "summary": summary,
        "download_url": f"/download/{room_id}"  # frontend should call this
    })

@app.route("/download/<room_id>", methods=["GET"])
def download_summary(room_id):
    pdf_path = os.path.join("meeting_reports", f"{room_id}_report.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"error": "No report for this room"}), 404

    # ✅ This actually streams the PDF file
    return send_file(pdf_path, as_attachment=True, download_name=f"{room_id}_report.pdf")

# ===== Ingest: Frames =====
@analyzer_bp.post("/ingest/frame")
def ingest_frame():
    data = request.get_json(force=True)
    room = data.get("roomId"); peer = data.get("peerId")
    frame_b64 = data.get("frame"); ts = data.get("timestamp") or int(time.time()*1000)
    if not (room and peer and frame_b64):
        return jsonify({"ok": False, "error":"missing fields"}), 400
    d = room_peer_dir(room, peer)
    b64 = re.sub(r"^data:image/\w+;base64,", "", frame_b64)
    img = base64.b64decode(b64)
    out = os.path.join(d, "frames", f"{ts}.jpg")
    with open(out, "wb") as f:
        f.write(img)
    return jsonify({"ok": True})

# ===== Ingest: Audio =====
@analyzer_bp.post("/ingest/audio")
def ingest_audio():
    room_id = request.form.get("roomId")
    peer_id = request.form.get("peerId")
    timestamp = request.form.get("timestamp")
    file = request.files.get("file")

    if not room_id or not peer_id or not file:
        return jsonify({"error": "Missing required fields"}), 400

    # Ensure folder exists
    dir_path = os.path.join("data", room_id, peer_id, "audio")
    os.makedirs(dir_path, exist_ok=True)

    # Save chunk
    filename = f"audio_{timestamp}.webm"
    save_path = os.path.join(dir_path, filename)
    file.save(save_path)

    # Debug log
    size_kb = os.path.getsize(save_path) / 1024
    print(f"🎧 Received audio chunk from {peer_id} in room {room_id}")
    print(f"   → File: {filename} ({size_kb:.1f} KB)")
    print(f"   → Saved at: {save_path}")

    # Here you can call AssemblyAI for realtime transcription
    # (or just simulate with dummy response while testing)
    # transcripts = []  # placeholder
    # status = "received"

    return jsonify({
        "ok": True
    })

def write_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)

def analyze_peer_frames(peer_dir):
    frames_dir = os.path.join(peer_dir, "frames")
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")],
    key=lambda x: int(x.replace("frame_", "").split(".")[0]))
    if not files: return []

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
    events = []
    def logger(t, ev, desc):
        ts = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        events.append((ts, ev, desc, ""))

    logic = EngagementLogic(logger)
    hand_y_window = deque(maxlen=HAND_MOVEMENT_WINDOW_FRAMES)
    last_ts_s = None

    for fname in files:
        ts_ms = int(os.path.splitext(fname.replace("frame_", ""))[0])
        ts_s = ts_ms/1000.0
        last_ts_s = ts_s if last_ts_s is None else max(last_ts_s + 1.0/VIDEO_FPS_TARGET, ts_s)

        img_bgr = cv2.imread(os.path.join(frames_dir, fname))
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        face_res = mp_face_mesh.process(img_rgb)
        hand_res = mp_hands.process(img_rgb)

        yaw = pitch = 0.0
        is_focused = False
        ear = 1.0
        mar = 0.0

        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark

            def coords(idxs):
                return [(int(lm[i].x*W), int(lm[i].y*H)) for i in idxs]

            left_idx = [362,385,387,263,373,380]
            right_idx= [33,160,158,133,153,144]
            mouth_idx= [61,81,13,311,402,14]

            ear = (get_eye_aspect_ratio(coords(left_idx)) + get_eye_aspect_ratio(coords(right_idx))) / 2.0
            mar = get_mouth_aspect_ratio(coords(mouth_idx))

            p, y, _ = head_pose(lm, img_rgb.shape)
            pitch, yaw = p, y
            is_focused = (abs(yaw) <= ATTENTION_YAW_THRESHOLD) and (abs(pitch) >= PITCH_FOCUSED_MIN_ABS_THRESHOLD)

        logic.update_attention(is_focused, yaw, pitch, last_ts_s)
        logic.register_blink(ear, last_ts_s)
        logic.register_yawn(mar, last_ts_s)

        is_raised = False
        hand_std = None
        if hand_res.multi_hand_landmarks:
            for h in hand_res.multi_hand_landmarks:
                wrist_y = h.landmark[0].y
                eye_y = ((lm[33].y + lm[263].y)/2) if face_res.multi_face_landmarks else 0.5
                if wrist_y < eye_y * HAND_RAISE_Y_THRESHOLD_FACTOR:
                    is_raised = True
                hand_y_window.append(wrist_y)
            if len(hand_y_window) >= 10:
                hand_std = float(np.std(hand_y_window))
        logic.register_hand(is_raised, hand_std, last_ts_s)

    mp_face_mesh.close()
    mp_hands.close()
    return events

def decode_webm_to_wav_bytes(path_webm, target_sr=AUDIO_SAMPLE_RATE):
    cmd = [
        "ffmpeg","-y","-i", path_webm,
        "-ac","1","-ar",str(target_sr),
        "-f","wav","-"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.stdout

def transcribe_peer_audio(peer_dir, whisper_model):
    audio_dir = os.path.join(peer_dir, "audio")
    # files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".webm")], key=lambda x: int(os.path.splitext(x.replace("audio_", ""))[0]))
    # if not files: return []

    files = sorted(
            [f for f in os.listdir(audio_dir) if f.startswith("audio_") and f.endswith(".webm")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

    pcm_all = []
    total_samples = 0
    for f in files:
        wav_bytes = decode_webm_to_wav_bytes(os.path.join(audio_dir, f))
        if not wav_bytes:
            print(f"⚠️ Skipping empty audio {f}")
            continue
        buf = io.BytesIO(wav_bytes); buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32")
        if sr != AUDIO_SAMPLE_RATE:
            audio = audio.astype(np.float32)
        pcm_all.append(audio)
        total_samples += len(audio)

    if not pcm_all: return []

    full_audio = np.concatenate(pcm_all).astype(np.float32)
    session_start = time.time() - float(len(full_audio))/AUDIO_SAMPLE_RATE

    segments, _ = whisper_model.transcribe(
        io.BytesIO(write_wav_bytes(full_audio, AUDIO_SAMPLE_RATE)),
        vad_filter=True, language="en", beam_size=5
    )

    rows = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            ts = datetime.fromtimestamp(session_start + seg.start).strftime("%H:%M:%S")
            rows.append((ts, text))
    return rows

def write_wav_bytes(audio, sr):
    b = io.BytesIO()
    sf.write(b, audio, sr, format="WAV")
    b.seek(0)
    return b.read()

def attach_speech_context(events, transcriptions):
    def to_sec(hms):
        t = datetime.strptime(hms, "%H:%M:%S").time()
        return t.hour*3600 + t.minute*60 + t.second
    trans_list = [(to_sec(ts), txt) for ts, txt in transcriptions]
    out = []
    for ts, ev, desc, _ in events:
        es = to_sec(ts)
        prior = ""
        best = -1
        for t, txt in trans_list:
            if t <= es and t > best:
                best = t; prior = txt
        out.append((ts, ev, desc, prior))
    return out

# ===== Real-Time Transcription API using AssemblyAI =====
def store_transcript(room_id, text, peer_id=None, ts=None, final=False):
    if not text.strip():
        print("⚠️ Ignoring empty transcript")
        return
    transcript_logs[room_id].append({
        "peerId": peer_id,
        "text": text.strip(),
        "final": final,
        "timestamp": ts or datetime.now().strftime("%H:%M:%S")
    })
    # print(transcript_logs[room_id])

def start_streaming_session(room_id, api_key):
    """
    Start one AssemblyAI streaming session per room.
    """
    if room_id in sessions:  # 👈 already running
        return sessions[room_id]["client"]

    transcripts = []
    audio_queue = Queue()

    client = StreamingClient(
        StreamingClientOptions(
            api_key=api_key,
            api_host="streaming.assemblyai.com",
        )
    )

    # --- Event Handlers ---
    def on_begin(client, event: BeginEvent):
        print(f"🔗 Session started for room={room_id}: {event.id}")

    def on_turn(client, event: TurnEvent):
        # ignore empty transcripts
        if not getattr(event, "transcript", None):
            return

        # sane defaults
        peer_id = sessions[room_id].get("last_peer", "Unknown")

        # use attributes safely
        turn_order = getattr(event, "turn_order", None)
        is_formatted = bool(getattr(event, "turn_is_formatted", False))
        is_final = bool(getattr(event, "end_of_turn", False))

        # dedupe key per room
        seen = sessions[room_id].setdefault("seen_turns", set())
        dedupe_key = (turn_order, is_formatted)

        if dedupe_key in seen:
            # already handled this exact message (avoid duplicates)
            return
        seen.add(dedupe_key)

        # Save every transcript into the in-memory session list (for UI if you want).
        sessions[room_id]["transcripts"].append({
            "peerId": peer_id,
            "text": event.transcript,
            "final": is_final,
            "formatted": is_formatted,
            "turn_order": turn_order,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        # ONLY persist the nicely formatted final (punctuated) into transcript_logs
        # This avoids double writes (raw final + formatted final)
        if is_final and is_formatted:
            # store_transcript accepts peer_id and ts (you already have this signature)
            store_transcript(room_id, event.transcript, peer_id=peer_id, ts=datetime.now().strftime("%H:%M:%S"), final=True)
            print(f"📝 {room_id} [{peer_id}] stored FINAL (formatted): {event.transcript}")
        else:
            # optionally print interims / raw finals
            kind = "final" if is_final else "interim"
            # print(f"📝 {room_id} [{peer_id}] {kind} (formatted={is_formatted}): {event.transcript}")

    def on_terminated(client, event: TerminationEvent):
        print(f"🔚 Session terminated: {event.audio_duration_seconds} sec processed")

    def on_error(client, error: StreamingError):
        print(f"❌ Error in {room_id}: {error}")

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    # --- Streaming loop in a background thread ---
    def streaming_loop():
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
            )
        )

        def generator():
            while True:
                try:
                    chunk = audio_queue.get(timeout=1)
                except Empty:
                    continue
                if chunk is None:
                    break
                print(f"Sending chunk of {len(chunk)} bytes")
                yield chunk


        client.stream(generator())
        client.disconnect(terminate=True)

    thread = threading.Thread(target=streaming_loop, daemon=True)
    thread.start()
    
    # Update session storage to include per-peer ffmpeg processes
    sessions[room_id] = {
    "client": client,
    "queue": audio_queue,
    "transcripts": transcripts,
    "last_peer": "Unknown",
    "seen_turns": set(),
    "peers": {}  # 👈 add a map for per-peer ffmpeg processes
    }


    return client

# def ensure_peer_ffmpeg(room_id, peer_id):
#     room = sessions[room_id]
#     if peer_id not in room["peers"]:
#         room["peers"][peer_id] = get_or_start_peer_ffmpeg(room_id, peer_id)
#     return room["peers"][peer_id]

def get_or_start_peer_ffmpeg(room_id, peer_id):
    room = sessions[room_id]
    peers = room["peers"]

    if peer_id in peers and peers[peer_id].poll() is None:
        return peers[peer_id]  # already running

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "quiet",
            "-f", "webm",
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "pipe:1"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        bufsize=10**6
    )

    # reader thread that pushes PCM into the shared queue
    def reader():
        CHUNK_SIZE = 6400
        buffer = b""
        print(ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else "No stderr")

        while True:
            data = ffmpeg_proc.stdout.read(CHUNK_SIZE)
            if not data:
                break
            
            buffer += data
            while len(buffer) >= CHUNK_SIZE:
                chunk, buffer = buffer[:CHUNK_SIZE], buffer[CHUNK_SIZE:]
                room["queue"].put(chunk)
                sessions[room_id]["last_peer"] = peer_id  # 👈 track who spoke last

    threading.Thread(target=reader, daemon=True).start()
    peers[peer_id] = ffmpeg_proc
    print(f"🚀 Started ffmpeg for peer {peer_id} in room {room_id}")
    return ffmpeg_proc

def push_audio_chunk(room_id, chunk_b64, peer_id):
    if room_id not in sessions:
        print(f"⚠️ No active session for {room_id}")
        return

    proc = get_or_start_peer_ffmpeg(room_id, peer_id)

    try:
        chunk = base64.b64decode(chunk_b64)
        proc.stdin.write(chunk)
        proc.stdin.flush()
    except BrokenPipeError:
        print(f"❌ Broken pipe for peer {peer_id} in room {room_id}")
    except Exception as e:
        print(f"❌ Error writing to ffmpeg for peer {peer_id}: {e}")


def stop_session(room_id):
    """
    Stop session gracefully.
    """
    if room_id in sessions:
        # stop the streaming loop
        sessions[room_id]["queue"].put(None)

        # terminate all peer ffmpeg procs
        for peer_id, proc in sessions[room_id].get("peers", {}).items():
            if proc and proc.poll() is None:
                proc.terminate()
                print(f"🛑 Killed ffmpeg for peer {peer_id} in room {room_id}")

        del sessions[room_id]
        print(f"🛑 Session stopped for room {room_id}")

@app.route("/start_session/<room_id>", methods=["POST"])
def start_session(room_id):
    if room_id in sessions:
        return jsonify({"status": "already running"})
    start_streaming_session(room_id, ASSEMBLYAI_API_KEY)
    return jsonify({"status": "started", "room": room_id})

@app.route("/transcribe/realtime/<room_id>", methods=["POST"])
def transcribe_realtime(room_id):
    data = request.json
    chunk_b64 = data.get("audio")
    peer_id = data.get("peerId", "unknown")  # ✅ receive peerId from Node

    if not chunk_b64 or not peer_id:
        return jsonify({"error": "missing audio or peerId"}), 400
    
    # Save peer for this room
    if room_id not in sessions:
        return jsonify({"error": f"Session {room_id} not active"}), 400
    
    sessions[room_id]["last_peer"] = peer_id

    push_audio_chunk(room_id, chunk_b64, peer_id)  # pass along peerId
    return jsonify({"status": "ok"})

@app.route("/get_transcripts/<room_id>", methods=["GET"])
def get_transcripts(room_id):
    if room_id not in sessions:
        return jsonify({"error": "no session"}), 404

    transcripts = transcript_logs.get(room_id, [])
    print(f"Returning {len(transcripts)} transcripts for room {room_id}")
    return jsonify({"transcripts": transcripts})

@app.route("/stop_session/<room_id>", methods=["POST"])
def stop(room_id):
    stop_session(room_id)
    return jsonify({"status": "stopped"})

# ===== Real-Time Frame Analysis API =====
@app.route("/analyze/realtime", methods=["POST"])
def analyze_realtime():
    data = request.get_json(force=True)
    room = data.get("roomId")
    peer = data.get("peerId")
    frame_b64 = data.get("frame")
    ts_ms = data.get("timestamp") or int(time.time() * 1000)
    if not (room and peer and frame_b64):
        return jsonify({}), 400

    # Decode frame
    b64 = re.sub(r"^data:image/\w+;base64,", "", frame_b64)
    img = base64.b64decode(b64)
    img_array = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({}), 400

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1)

    eventType, description = None, None

    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        face_res = mp_face_mesh.process(img_rgb)
        hand_res = mp_hands.process(img_rgb)

        # --- Default eye_y (mid-frame) ---
        eye_y = 0.5

        # --- Hand raise detection first ---
        if hand_res.multi_hand_landmarks:
            for h in hand_res.multi_hand_landmarks:
                wrist_y = h.landmark[0].y
                if wrist_y < eye_y * HAND_RAISE_Y_THRESHOLD_FACTOR:
                    eventType = "HandRaise"
                    description = "Hand raised (possible question)"

        # --- Face detection second ---
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark

            # Update eye_y using facial landmarks
            eye_y = (lm[33].y + lm[263].y) / 2

            # --- Head pose for distraction ---
            p, y, _ = head_pose(lm, img_rgb.shape)
            focused = (abs(y) <= ATTENTION_YAW_THRESHOLD) and (abs(p) >= PITCH_FOCUSED_MIN_ABS_THRESHOLD)
            if not focused:
                eventType = "Distraction"
                description = "Looks distracted (head pose)"

            if focused:
                eventType = "Focus"
                description = "Looks focused (head pose)"

            # --- Eye & mouth for fatigue ---
            left_idx = [362, 385, 387, 263, 373, 380]
            right_idx = [33, 160, 158, 133, 153, 144]
            mouth_idx = [61, 81, 13, 311, 402, 14]

            def coords(idxs):
                return [(int(lm[i].x * W), int(lm[i].y * H)) for i in idxs]

            ear = (get_eye_aspect_ratio(coords(left_idx)) + get_eye_aspect_ratio(coords(right_idx))) / 2.0
            mar = get_mouth_aspect_ratio(coords(mouth_idx))

            if ear < EAR_THRESHOLD:
                eventType = "Fatigue"
                description = "May be drowsy (blink detected)"
            if mar > MAR_THRESHOLD:
                eventType = "Fatigue"
                description = "May be yawning (possible fatigue)"

    finally:
        mp_face_mesh.close()
        mp_hands.close()

    if eventType:
        event = {
            "peerId": peer,
            "eventType": eventType,
            "description": description,
            "timestamp": datetime.fromtimestamp(ts_ms / 1000.0).strftime("%H:%M:%S")
        }
        # Store event immediately
        engagement_events[room].append(event)

        return jsonify(event)

    return jsonify({})

# ===== Flask App and Blueprint Registration =====
app.register_blueprint(analyzer_bp)


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0", debug=True, use_reloader=False)