# video_ingest.py
# Phase 2 - Video Ingestion & Preprocessing
# Loads a Hudl video, checks its quality, and extracts every frame as an image

import cv2          # OpenCV - handles all video and image operations
import os           # lets us create folders and build file paths
import json         # FIX: added for manifest saving
import numpy as np  # NumPy - used for math operations on image data


# ─────────────────────────────────────────────
# FUNCTION 1 - measure_sharpnessOfVideoFrame
# ─────────────────────────────────────────────
def measure_sharpnessOfVideoFrame(frame):
    videoFramesToGrayFrames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness_score = cv2.Laplacian(videoFramesToGrayFrames, cv2.CV_64F).var()
    return sharpness_score


# ─────────────────────────────────────────────
# FUNCTION 2 - videoQualityChecker
# ─────────────────────────────────────────────
# FIX: added reported_total_frames, decoded_frames, decode_warnings
# Research: "treat metadata as hints not truth — trust what you decoded"

def videoQualityChecker(video_path):
    capturedVideo = cv2.VideoCapture(video_path)

    # BUG WAS: had stray "..." on this line which caused syntax error
    # FIX: removed the stray ellipsis
    if not capturedVideo.isOpened():
        print("ERROR: Video file failed to open:", video_path)
        return None

    success, frame = capturedVideo.read()

    video_width            = int(capturedVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height           = int(capturedVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps              = capturedVideo.get(cv2.CAP_PROP_FPS)

    # FIX: renamed to reported_total_frames — this is what the file CLAIMS
    # we cannot trust this number — codec quirks can make it wrong
    reported_total_frames  = int(capturedVideo.get(cv2.CAP_PROP_FRAME_COUNT))

    sharpness = measure_sharpnessOfVideoFrame(frame) if success else 0

    capturedVideo.release()

    # FIX: count frames we ACTUALLY decoded vs what was reported
    # research: "decoded_frames is truth, reported_total_frames is a hint"
    decoded_frames = 0
    decode_warnings = []

    verify_cap = cv2.VideoCapture(video_path)
    while True:
        ok, _ = verify_cap.read()
        if not ok:
            break
        decoded_frames += 1
    verify_cap.release()

    # FIX: check if reported vs decoded are significantly different
    frame_difference = abs(reported_total_frames - decoded_frames)
    if frame_difference > 10:
        warning = (f"⚠️  Frame count mismatch: "
                   f"reported {reported_total_frames} "
                   f"but decoded {decoded_frames}")
        decode_warnings.append(warning)
        print(warning)

    # FIX: added reported_total_frames, decoded_frames, decode_warnings
    # total_frames now uses decoded_frames — the real number
    video_quality_info = {
        "width":                   video_width,
        "height":                  video_height,
        "fps":                     video_fps,
        "reported_total_frames":   reported_total_frames,
        "decoded_frames":          decoded_frames,
        "total_frames":            decoded_frames,   # trust decoded not reported
        "sharpness_score":         round(sharpness, 2),
        "needs_upscaling":         video_width < 1280 or sharpness < 100.0,
        "decode_warnings":         decode_warnings
    }

    return video_quality_info


# ─────────────────────────────────────────────
# FUNCTION 3 - extract_video_frames
# ─────────────────────────────────────────────
def extract_video_frames(video_path, output_folder="data/frames"):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video:", video_path)
        return 0

    # FIX: read actual fps from video — never hardcode
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # BUG WAS: "acutal_fps" — typo, Python would crash with NameError
    # FIX: corrected spelling to actual_fps
    # also guard against 0 fps to prevent ZeroDivisionError crash
    if actual_fps <= 0:
        actual_fps = 30.0
        print("  ⚠️  Warning: FPS reported as 0 — defaulting to 30fps")

    frame_count    = 0
    decode_errors  = 0
    max_decode_errors = 10

    # FIX: use actual_fps in progress message — not hardcoded 10
    # BUG WAS: "rouund" — typo, would crash with NameError
    seconds_per_300 = round(300 / actual_fps, 1)
    print(f"Every 300 frames = ~{seconds_per_300} seconds at {actual_fps} fps")

    while True:
        success, frame = cap.read()

        # FIX: differentiate clean end of file vs corrupt decode error
        # BUG WAS: treated ALL failures as end of file with just "break"
        # research: "differentiate decode error vs end of file for logging"
        if not success:
            if frame is None:
                # clean end of file — normal exit
                break
            else:
                # corrupt frame — log it and try to continue
                decode_errors += 1
                print(f"  ⚠️  Decode error at frame {frame_count} "
                      f"({decode_errors} total errors)")
                if decode_errors > max_decode_errors:
                    print(f"  ❌ Too many decode errors — stopping early")
                    break
                continue

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # BUG WAS: progress print block was OUTSIDE the while loop
        # FIX: indented inside loop so it actually runs during extraction
        if frame_count % 300 == 0:
            elapsed_seconds = round(frame_count / actual_fps, 1)
            print(f"  Extracted {frame_count} frames "
                  f"(~{elapsed_seconds} seconds) so far...")

    cap.release()

    # FIX: use actual_fps instead of hardcoded 10
    # BUG WAS: frame_count / 10 — only correct at exactly 30fps
    total_seconds = round(frame_count / actual_fps, 1)

    # BUG WAS: print("Total of :", {frame_count}...) — curly braces make a SET
    # FIX: use f-string for clean number output
    print(f"Total of: {frame_count} frames = ~{total_seconds} seconds of footage")
    print(f"Done! Extracted {frame_count} total frames → saved to: {output_folder}")

    return frame_count


# ─────────────────────────────────────────────
# FUNCTION 4 - save_video_manifest
# ─────────────────────────────────────────────
# FIX: new function — saves quality info to JSON file
# every phase reads this instead of re-analyzing the video
# research: "store manifest per video for traceability"

def save_video_manifest(video_quality_info, output_folder="data/frames"):

    os.makedirs(output_folder, exist_ok=True)

    manifest_path = os.path.join(output_folder, "manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(video_quality_info, f, indent=4)

    print(f"  ✅ Manifest saved → {manifest_path}")
    return manifest_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_video_path = "data/film/vidvid.mp4"

    print("=== GridironIQ Film Processor ===")
    print("Checking video quality...\n")

    video_quality = videoQualityChecker(test_video_path)

    if video_quality:
        print("Video Quality Report:")
        print(f"  Resolution        : {video_quality['width']} x {video_quality['height']}")
        print(f"  Frame Rate        : {video_quality['fps']} fps")

        # FIX: now prints both reported and decoded frame counts
        print(f"  Reported Frames   : {video_quality['reported_total_frames']}")
        print(f"  Decoded Frames    : {video_quality['decoded_frames']}")
        print(f"  Sharpness Score   : {video_quality['sharpness_score']}")
        print(f"  Needs Upscaling   : {video_quality['needs_upscaling']}")

        # FIX: print any decode warnings found
        if video_quality['decode_warnings']:
            print("\n⚠️  Warnings:")
            for warning in video_quality['decode_warnings']:
                print(f"  {warning}")

        # FIX: save manifest for other phases to read
        print("\nSaving video manifest...")
        save_video_manifest(video_quality)

        print("\nExtracting frames...")
        extract_video_frames(test_video_path)