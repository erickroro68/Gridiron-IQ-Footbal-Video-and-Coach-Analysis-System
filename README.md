
     🏈 GridironIQ — AI Football Film Intelligence




📌 Overview
============================

 -  GridironIQ is an AI-driven football film analysis system designed to help coaches spend less time fighting the film and more time coaching the game. Built with Python, YOLOv8, and ByteTrack, it detects formations, tracks players, analyzes tendencies, and simulates plays through a multi-phase pipeline that turns raw film into data-driven insight. The goal is simple: help coaches break down opponents faster, catch details that are easy to miss, self-scout more honestly, and give smaller staffs access to a level of football intelligence that usually takes far more time, money, and manpower.

     -Designed with a modular, multi‑phase architecture,             making it easy to extend with new models, add custom           team‑specific logic, and integrate future features             like real‑time sideline analysis or automated scouting         reports.

     -Transforms raw game film into structured, searchable           data, giving coaches instant access to formations,             tendencies, and player behavior without hours of               manual tagging.

     -Built for real football environments, from high‑school         programs with small staffs to college teams looking for        an edge, delivering analytics that normally require           expensive systems and large analyst crews


     **Built for:**

     - High school programs

     - College (D3–D1)

     - Advanced scouting workflows




⚡ What GridironIQ Does
============================

- Detects players using YOLOv8 + Soft-NMS

-Tracks players across frames (ByteTrack – upcoming)

-Classifies formations (offense & defense)

-Computes opponent tendencies

-Simulates plays using probabilistic models

-Generates automated scouting reports

-Builds toward real-time sideline analysis




🧠 Why It’s Different
============================


-Unlike tools like Hudl (manual tagging), GridironIQ:

-Automates entire film breakdown

-Runs 1000+ simulations per play

-Flags hidden tendencies coaches miss

-Bridges film → analytics → decision-making




🏗️ System Architecture (Phases)



✅ Completed

-Phase 1: Environment setup

-Phase 2: Video ingestion

-Phase 3: Frame upscaling

-Phase 4 (Partial): Player detection + Soft-NMS




🚧 In Progress




Field validation (field_checker)

Perspective correction

Bounding box scoring




🔜 Upcoming




Phase 5: Player tracking (ByteTrack)

Phase 6: Snap detection

Phase 7: Field homography

Phase 8: Formation recognition

Phase 9+: AI analytics + simulation engine




🧱 Tech Stack
============================

Python 3.10+

NumPy

OpenCV

Ultralytics YOLOv8

PyTorch

Pipeline / Backend

FastAPI (API layer)

SQLAlchemy (data storage)

MLflow (experiment tracking)

ML / Vision Extensions

ByteTrack / BoT-SORT

OCR (Pytesseract)

Homography (OpenCV)




⚙️ Installation
============================


1. Create Virtual Environment
python -m venv .venv
Activate:

Windows:

.\.venv\Scripts\activate

Mac/Linux:

source .venv/bin/activate

2. Install Core Dependencies

pip install -r requirements.txt

5. Install PyTorch (IMPORTANT)



⚠️ PyTorch depends on your GPU
============================


CPU:
pip install torch torchvision
GPU (example CUDA 12.1):
pip install torch==2.7.0+cu121 torchvision==0.22.0+cu121 \
-f https://download.pytorch.org/whl/torch_stable.html
📦 requirements.txt (Core)
numpy==2.4.3
opencv-python==4.13.0.92
ultralytics==8.4.23

fastapi==0.135.1
uvicorn==0.42.0

SQLAlchemy==2.0.48
alembic==1.18.4

mlflow==3.10.1
prometheus-client==0.24.1
opentelemetry-api==1.40.0
opentelemetry-sdk==1.40.0

streamlit==1.55.0

lap==0.5.13
cython-bbox==0.1.5
pytesseract==0.3.13
🧩 Optional Requirements
Dev tools
pytest
black
flake8
mypy
pip-tools
ML / Advanced
scikit-learn
faiss-cpu
albumentations
matplotlib
🧪 Minimal Test
python -c "import numpy, cv2, ultralytics; print('OK')"

============================


YOLO test:
============================

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict("test.jpg")



🔁 Reproducibility Checklist
============================

Use Python 3.10–3.12

Always use a virtual environment

Pin all dependencies

Match CUDA ↔ PyTorch versions

Document external installs (Tesseract, models)




🤖 Tracking Algorithms Comparison
============================

Tracker	Strength	Weakness

ByteTrack	Fast, handles occlusion	No appearance tracking

BoT-SORT	Most accurate	Slower

DeepSORT	Simple	Outdated


👉 Default: ByteTrack

👉 Advanced: BoT-SORT




🚀 Future Research Directions
============================


Field line detection (hash marks + yard lines)

Jersey number OCR

Route classification (WR tracking)

Blitz detection (pre-snap movement)

Weather-aware model tuning

Real-time sideline deployment




🎯 Vision

============================

GridironIQ aims to become:

“The AI offensive & defensive coordinator assistant.”

Turning raw film into:

Insights

Predictions

Winning decisions
