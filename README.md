# GridironIQ — AI Football Film Intelligence

GridironIQ is an AI-powered football film analysis system designed to automate what coaching staffs traditionally do manually. The platform ingests raw game film, detects and tracks all 22 players, identifies formations and coverages, analyzes opponent tendencies, and generates data-driven insights to support game planning at the high school and D3 levels.

## What GridironIQ Does
- Detects players using YOLOv8 with production-grade Soft-NMS  
- Tracks players across frames with ByteTrack  
- Maps field coordinates using homography for real-world positioning  
- Classifies offensive formations, defensive fronts, motions, and personnel  
- Computes opponent tendencies with Bayesian modeling  
- Simulates plays using Monte Carlo methods to rank optimal calls  
- Generates automated game plans and self-scout reports  
- Provides a foundation for real-time sideline analysis

## Why It’s Different
Unlike Hudl, which surfaces data for coaches to interpret manually, GridironIQ closes the loop by:
- Recommending plays based on opponent tendencies  
- Running 1000+ simulations per play with uncertainty modeling  
- Flagging your own predictable tendencies  
- Integrating fatigue and injury‑risk metrics  
- Offering advanced analytics to programs without NFL-level resources  

## Tech Stack
Python 3.13 • YOLOv8 • OpenCV • PyTorch • ByteTrack • NumPy • Pandas • Flask • SQLite • ReportLab

## Current Status
**Completed:** Ingestion, preprocessing, upscaling  
**In Progress:** Player detection pipeline (YOLOv8 + Soft-NMS + smoothing)  
**Next:** Tracking → Formation recognition → Tendency engine → Simulation → Game plan generator → Dashboard → Sideline mode

## Getting Started
