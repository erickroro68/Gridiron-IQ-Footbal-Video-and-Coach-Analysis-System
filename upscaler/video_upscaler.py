# video_upscaler.py
# Phase 3 - Video Upscaling
# Takes frames from Phase 2 and makes them bigger and clearer
# Only runs if Phase 2 flagged needs_upscaling as True

import cv2   # handles image loading, resizing, saving
import os    # file path and folder operations
import json  # FIX: added to read manifest from Phase 2


# ─────────────────────────────────────────────
# FUNCTION 1 - load_video_manifest
# ─────────────────────────────────────────────
# FIX: new function — reads the manifest Phase 2 saved
# instead of re-analyzing the video from scratch
# research: "each phase reads manifest instead of re-analyzing"

def load_video_manifest(frames_folder="data/frames"):
    manifest_path = os.path.join(frames_folder, "manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)

    # no manifest found — warn the user
    print("  ⚠️  No manifest found — run video_ingest.py first")
    return None


# ─────────────────────────────────────────────
# FUNCTION 2 - frame_upscaler
# ─────────────────────────────────────────────
# Takes ONE frame and returns it upscaled
# scale_factor=2 means double the resolution by default

def frame_upscaler(frame, scale_factor=2):

    # frame.shape returns (height, width, channels)
    # BUG WAS: comments had shape[0] and shape[1] swapped in description
    # FIX: shape[0] = HEIGHT (rows), shape[1] = WIDTH (columns)
    og_height = frame.shape[0]   # height = rows
    og_width  = frame.shape[1]   # width  = columns

    new_width  = og_width  * scale_factor
    new_height = og_height * scale_factor

    upscaled_frame = cv2.resize(
        frame,
        dsize=(new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    return upscaled_frame


# ─────────────────────────────────────────────
# FUNCTION 3 - upscale_process
# ─────────────────────────────────────────────
# Loops through ALL frames and calls frame_upscaler on each one
# Only runs if needs_upscaling flag is True from Phase 2

def upscale_process(
    input_folder="data/frames",
    output_folder="data/frames_upscaled",
    scale_factor=2,
    needs_upscaling=True,
    actual_fps=30.0   # FIX: accept fps so we never hardcode it
):
    # BUG WAS: print referenced "frame" variable that doesn't exist here
    # FIX: removed frame reference from skip message
    if not needs_upscaling:
        print("Video quality is good — upscaling not needed, skipping.")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    print("Starting upscale process...")

    frame_files = sorted([
        f for f in os.listdir(input_folder) if f.endswith(".jpg")
    ])

    total_frames = len(frame_files)

    if total_frames == 0:
        print("ERROR: No frames found in", input_folder)
        print("Run video_ingest.py first.")
        return 0

    print(f"Found {total_frames} frames to upscale at {scale_factor}x resolution")

    # use actual_fps instead of hardcoded 30
    # BUG WAS: "Every 300 frames = ~10 seconds" — hardcoded 30fps assumption
    seconds_per_300 = round(300 / actual_fps, 1)
    print(f"Every 300 frames = ~{seconds_per_300} seconds\n")

    upscaled_count = 0

    for i, filename in enumerate(frame_files):

        input_path  = os.path.join(input_folder,  filename)
        output_path = os.path.join(output_folder, "UPSCALED_" + filename)

        loaded_frame = cv2.imread(input_path)

        if loaded_frame is None:
            print(f"  Skipping unreadable frame: {filename}")
            continue

        upscaled_frame = frame_upscaler(loaded_frame, scale_factor)
        cv2.imwrite(output_path, upscaled_frame)
        upscaled_count += 1

        if upscaled_count % 300 == 0:
            percent = round((upscaled_count / total_frames) * 100, 1)
            # FIX: use actual_fps for elapsed seconds
            elapsed = round(upscaled_count / actual_fps, 1)
            print(f"  Upscaled {upscaled_count} / {total_frames} "
                  f"frames ({percent}% done ~{elapsed}s)...")

    # FIX: use actual_fps instead of hardcoded 30
    # BUG WAS: round(upscaled_count / 30, 1) — hardcoded fps assumption
    total_seconds = round(upscaled_count / actual_fps, 1)
    print(f"\nDone! {upscaled_count} frames upscaled → saved to: {output_folder}")
    print(f"Approx video length: ~{total_seconds} seconds")

    return upscaled_count


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    import sys
    sys.path.append(".")

    from film_processor.video_ingest import videoQualityChecker

    video_path = "data/film/vidvid.mp4"

    print("=== GridironIQ Video Upscaler ===\n")

    # FIX: read manifest first — don't re-analyze video
    manifest = load_video_manifest()

    if manifest:
        # FIX: get actual fps from manifest — never hardcode
        actual_fps = manifest.get("fps", 30.0)
        print(f"  Loaded manifest — fps: {actual_fps}")
        print(f"  Needs Upscaling : {manifest['needs_upscaling']}\n")

        upscale_process(
            needs_upscaling=manifest["needs_upscaling"],
            actual_fps=actual_fps
        )
    else:
        # fallback — run quality checker directly if no manifest
        quality = videoQualityChecker(video_path)
        if quality:
            actual_fps = quality.get("fps", 30.0)
            print(f"  Needs Upscaling : {quality['needs_upscaling']}\n")
            upscale_process(
                needs_upscaling=quality["needs_upscaling"],
                actual_fps=actual_fps
            )