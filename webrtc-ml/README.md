# Web Meeting Attention Tracker

An experimental **AI-powered web meeting analysis system** that combines real-time facial landmark tracking with speech transcription to analyze participant attention during virtual meetings.

## Overview

The system uses **MediaPipe Face Mesh** to track facial landmarks and analyzes their relative positions and movements to estimate attention-related states such as:

- Focused
- Drowsy
- Distracted
- Yawning
- Other observable attention cues

At the same time, meeting audio is processed through **AssemblyAI** for real-time speech transcription.

The goal was to explore how multiple streams of information—**facial behaviour and meeting speech**—could be combined to generate useful reports for meeting stakeholders.

## Architecture

```text
Webcam
   ↓
MediaPipe Face Landmarks
   ↓
Facial Feature Analysis
   ↓
Attention State
   │
   ├── Focused
   ├── Drowsy
   ├── Distracted
   └── Yawning

Microphone
   ↓
AssemblyAI
   ↓
Real-time Transcript
   │
   └──────────────┐
                  ↓
          Meeting Analysis
                  ↓
              Report
````

## Tech Stack

* **Python**
* **MediaPipe**
* **OpenCV**
* **JavaScript / Node.js**
* **AssemblyAI**
* **WebRTC**
* **HTML / CSS**

## Key Concepts

This experiment explored:

* Facial landmark detection
* Relative landmark geometry
* Eye and mouth movement analysis
* Attention-state classification
* Real-time video processing
* Real-time speech-to-text
* WebRTC-based communication
* Combining computer vision and speech data
* Generating meeting insights for stakeholders

## Project Structure

```text
webrtc-ml/
├── fonts/
├── public/
├── analyzer.py
├── server.js
├── test_transcription.py
├── sample.wav
├── requirements.txt
├── package.json
└── package-lock.json
```

## Purpose

This project was an early experiment in applying **computer vision and machine learning concepts to real-world human-computer interaction**.

It laid the groundwork for my continued exploration of:

**Computer Vision → Multimodal AI → Real-Time AI Systems**

---

**Status:** Experimental / Prototype
