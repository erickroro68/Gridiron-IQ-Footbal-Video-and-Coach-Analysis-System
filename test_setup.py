# test_setup.py
# This file verifies all our Phase 1 libraries installed correctly

# Import each library we installed
import cv2          # OpenCV - for video processing
import numpy as np  # NumPy - for math and arrays
import pandas as pd # Pandas - for data tables
import flask        # Flask - for our API server
import torch        # PyTorch - for AI models

# Print versions so we know exactly what we have
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("PyTorch version:", torch.__version__)
print("Flask version:", flask.__version__)

print("\n✅ Phase 1 Complete — All libraries loaded successfully!")

