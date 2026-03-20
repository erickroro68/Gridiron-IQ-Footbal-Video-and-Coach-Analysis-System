# film_analyzer.py
# Phase 4 - Player Detection & Film Analysis
# Uses YOLOv8 to detect players in each frame
# Counts players on each side of the field
# Draws bounding boxes to visually verify detections

# BUG WAS: "from email.contentmanager import raw_data_manager" — wrong import
# BUG WAS: "from email.headerregistry import MessageIDHeader" — wrong import
# BUG WAS: "from wsgiref.handlers import format_date_time" — wrong import
# FIX: removed all unrelated imports — only import what we actually use

import cv2                    # handles image loading and drawing boxes
import os                     # file path and folder operations
from ultralytics import YOLO  # YOLOv8 - our player detection AI model


# ─────────────────────────────────────────────
# FUNCTION 1 - box_validation
# ─────────────────────────────────────────────
# Validates a bounding box before we use it
# Protects against corrupt detections crashing our analysis
# Returns True if valid, False if anything is wrong

def box_validation(bbox, frame_copy_width, frame_copy_height):

    # check 1 — must have exactly 4 coordinate values
    if len(bbox) != 4:
        return False

    # unpack all 4 coordinates in one line (tuple unpacking)
    x1, y1, x2, y2 = bbox

    # BUG WAS: x1 > x2 — should be >= to catch zero-width boxes too
    # FIX: x1 >= x2 catches both invalid AND zero-width boxes
    if x1 >= x2 or y1 >= y2:
        return False

    if x1 < 0 or y1 < 0:
        return False

    if x2 > frame_copy_width:
        return False

    if y2 > frame_copy_height:
        return False

    box_validity_check = True
    return box_validity_check


# ─────────────────────────────────────────────
# FUNCTION 2 - detecting_player_in_frame
# ─────────────────────────────────────────────
# Takes ONE frame and runs it through YOLOv8
# Filters out detections below 0.5 confidence
# Validates each bbox before adding to list
# Returns list of detected players with bbox and confidence

def detecting_player_in_frame(frame, pretrained_model):

    returned_players = []

    # get frame dimensions for bbox validation
    # frame.shape returns (height, width, channels)
    frame_copy_height = frame.shape[0]
    frame_copy_width  = frame.shape[1]

    # run frame through YOLOv8 — returns list of Result objects
    raw_video_data = pretrained_model(frame)

    for result in raw_video_data:

        for box in result.boxes:

            frame_confidence_result = float(box.conf[0])

            # BUG WAS: confidence check was AFTER coordinate extraction
            # FIX: check confidence FIRST — cheaper operation runs first
            # if confidence fails we never waste time extracting coordinates
            if frame_confidence_result < 0.5:
                continue

            # BUG WAS: box.xyxy[0].toList() — capital L crashes with AttributeError
            # FIX: box.xyxy[0].tolist() — lowercase l is the correct method name
            frame_coordinates = box.xyxy[0].tolist()

            # BUG WAS: missing colon at end of if statement — syntax error
            # FIX: added colon after box_validation call
            if not box_validation(frame_coordinates, frame_copy_width, frame_copy_height):
                continue

            # BUG WAS: dict keys were "bbox: " and "Confidence Grade: " with spaces/colons
            # FIX: clean keys "bbox" and "confidence" — must match exactly when accessed later
            returned_players.append({
                "bbox":       frame_coordinates,
                "confidence": frame_confidence_result
            })

    return returned_players


# ─────────────────────────────────────────────
# FUNCTION 3 - count_player_side
# ─────────────────────────────────────────────
# Splits frame down center and counts players on each side
# Returns dict with left_count, right_count, total_players

def count_player_side(frame_width, returned_players):

    # BUG WAS: function body was empty — just "split_frame_x_cords" with no logic
    # FIX: added complete implementation

    # split the frame width in half to get center dividing line
    split_frame_x_cord = frame_width / 2

    left_field_player_count  = 0
    right_field_player_count = 0

    for player in returned_players:

        bbox = player["bbox"]

        # find horizontal center point of this player's bounding box
        # (x1 + x2) / 2 gives us the middle x coordinate
        players_center_box_point = (bbox[0] + bbox[2]) / 2

        # compare player center to frame center
        if players_center_box_point < split_frame_x_cord:
            left_field_player_count  += 1
        else:
            right_field_player_count += 1

    formation_count = {
        "left_count":    left_field_player_count,
        "right_count":   right_field_player_count,
        "total_players": len(returned_players)
    }

    return formation_count


# ─────────────────────────────────────────────
# FUNCTION 4 - camera_angle_splits
# ─────────────────────────────────────────────
# Uses total player count to determine camera angle
# Capped at 22 — football max, anything over is refs or false detections

def camera_angle_splits(formation_count):

    # cap at 22 — we know football has max 22 players
    player_sum_total = min(formation_count["total_players"], 22)

    if player_sum_total >= 14:
        camera_angle_type = "wide"
    elif player_sum_total <= 10:
        camera_angle_type = "trench"
    else:
        camera_angle_type = "unknown"

    return camera_angle_type


# ─────────────────────────────────────────────
# FUNCTION 5 - total_frame_median
# ─────────────────────────────────────────────
# Sliding window median smoothing for camera angle stability
# Prevents single occluded frames from flipping camera angle

def total_frame_median(stacked_frames, current_players_in_frame_total,
                       max_stacked_frames=30):

    stacked_frames.append(current_players_in_frame_total)

    if len(stacked_frames) > max_stacked_frames:
        stacked_frames.pop(0)

    sorted_frames = sorted(stacked_frames)

    # BUG WAS: middle_index == len(...) — double equals is comparison not assignment
    # FIX: middle_index = len(...) — single equals assigns the value
    middle_index = len(sorted_frames) // 2

    windowList_frame_median = sorted_frames[middle_index]

    return stacked_frames, windowList_frame_median


# ─────────────────────────────────────────────
# FUNCTION 6 - draw_player_boxes
# ─────────────────────────────────────────────
# Draws blue bounding boxes and confidence scores on a copy of the frame
# Never modifies original — always draws on frame_copy

def draw_player_boxes(frame, returned_players):

    player_box_color = (255, 0, 0)   # blue in BGR format

    # copy original so we never modify the source frame
    frame_copy = frame.copy()

    for player in returned_players:

        bbox = player["bbox"]

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        # draw blue rectangle around detected player
        cv2.rectangle(
            frame_copy,
            (x1, y1),
            (x2, y2),
            player_box_color,
            2   # thickness = 2 pixels
        )

        # round confidence to 2 decimal places and convert to string
        box_confidence_text = str(round(player["confidence"], 2))

        # write confidence score 10 pixels above the top of the box
        # y1 - 10 places text just above the rectangle
        cv2.putText(
            frame_copy,
            box_confidence_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,              # font size
            player_box_color,
            1                 # text thickness
        )

    stored_drawn_frame = frame_copy

    return stored_drawn_frame


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    import sys
    sys.path.append(".")

    # BUG WAS: pretrained_model = YOLO(...) then called detecting_player_in_frame
    # on a variable "frame" that was never defined — would crash with NameError
    # FIX: load a real test frame from data/frames before calling detection

    # load model ONCE here — not inside the function
    # research: "loading model inside function = reload every frame = 1 hour wasted"
    pretrained_model = YOLO("yolov8n.pt")

    # load a test frame to verify detection works
    test_frame_path = "data/frames_upscaled"

    if os.path.exists(test_frame_path):
        frame_files = sorted([
            f for f in os.listdir(test_frame_path) if f.endswith(".jpg")
        ])

        if frame_files:
            # load first available frame
            first_frame = cv2.imread(
                os.path.join(test_frame_path, frame_files[0])
            )

            if first_frame is not None:
                print("=== GridironIQ Film Analyzer ===\n")
                print(f"Testing detection on: {frame_files[0]}")

                # run detection
                players = detecting_player_in_frame(first_frame, pretrained_model)
                print(f"Players detected: {len(players)}")

                # count sides
                counts = count_player_side(first_frame.shape[1], players)
                print(f"Left side:  {counts['left_count']}")
                print(f"Right side: {counts['right_count']}")
                print(f"Total:      {counts['total_players']}")

                # get camera angle
                angle = camera_angle_splits(counts)
                print(f"Camera angle: {angle}")

                # draw boxes and save result
                annotated = draw_player_boxes(first_frame, players)
                output_path = "data/frames_upscaled/detection_test.jpg"
                cv2.imwrite(output_path, annotated)
                print(f"\n✅ Detection test saved → {output_path}")
            else:
                print("❌ Could not load test frame")
        else:
            print("❌ No frames found — run video_ingest.py and video_upscaler.py first")
    else:
        print("❌ frames_upscaled folder not found — run Phase 2 and Phase 3 first")