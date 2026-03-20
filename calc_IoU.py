# soft_nms.py
# Phase 4 — Gaussian Soft-NMS (Production Version)
# Research: Bodla et al. ICCV 2017 + GridironIQ football tuning
#
# WHY THIS EXISTS:
# Hard NMS deletes overlapping boxes — kills real linemen standing 3ft apart
# Soft-NMS DECAYS scores instead — all players stay detectable
# Critical for trench play where 5 linemen physically touch each other
#
# RESEARCH SOURCES APPLIED:
# → Bodla et al. 2017: Gaussian decay formula s_i *= exp(-(IoU²/sigma))
# → Detectron2: separate prune_thresh vs final_thresh
# → MMCV: min_score=1e-3 for pruning, higher for final filter
# → FieldMOT/BroadTrack: register-then-track needs clean box inputs
# → GridironIQ research: adaptive sigma per camera angle
#
# PIPELINE POSITION (from research master blueprint):
# YOLOv8 → Soft-NMS → bbox_confidence fused score → ByteTrack
#
# KEY UPGRADES FROM NAIVE VERSION:
# 1. Vectorized IoU — 3-15x faster than Python loop (NumPy C-optimized)
# 2. Adaptive sigma — changes based on camera angle (wide vs trench)
# 3. Two thresholds — prune_thresh for speed, higher_pruning_threshold for quality
# 4. Hard kill switch — IoU >= 0.95 means true duplicate, delete immediately
# 5. Copy protection — NEVER mutates original input lists
# 6. Original index tracking — tracker needs to map back to source detections
# 7. Full input validation — catches bad geometry BEFORE IoU math runs

import numpy as np


# ─────────────────────────────────────────────
# FUNCTION 1 — compute_iou_vectorized
# ─────────────────────────────────────────────
# Computes IoU between ONE reference box and ALL remaining boxes at once
# Research: vectorized NumPy runs in C — 3-15x faster than Python for loop
# Called once per Soft-NMS iteration instead of N separate calculate_IoU calls
#
# HOW IT WORKS:
# Instead of looping: for box in boxes: iou = calculate_IoU(ref, box)
# NumPy computes ALL IoUs simultaneously using array math
#
# boxes[:, 0] means "column 0 of every row" = all x1 values at once

def compute_iou_vectorized(reference_box, remaining_boxes):
    # remaining_boxes is shape [N, 4] — N boxes each with 4 coordinates
    # reference_box is shape [4] — single box [x1, y1, x2, y2]

    # find intersection rectangle for ALL boxes simultaneously
    # np.maximum compares element-wise across the entire array
    intersection_left_x   = np.maximum(reference_box[0], remaining_boxes[:, 0])
    intersection_top_y    = np.maximum(reference_box[1], remaining_boxes[:, 1])
    intersection_right_x  = np.minimum(reference_box[2], remaining_boxes[:, 2])
    intersection_bottom_y = np.minimum(reference_box[3], remaining_boxes[:, 3])

    # clip to 0 — negative means no overlap at all
    intersection_width  = np.maximum(0.0, intersection_right_x  - intersection_left_x)
    intersection_height = np.maximum(0.0, intersection_bottom_y - intersection_top_y)

    # area of overlap for every box at once
    intersection_areas = intersection_width * intersection_height

    # area of reference box — single number
    reference_box_area = ((reference_box[2] - reference_box[0]) *
                          (reference_box[3] - reference_box[1]))

    # area of every remaining box — array of N numbers
    # BUG WAS: used & (bitwise AND) instead of * (multiply) — completely wrong math
    # FIX: use * for multiplication
    remaining_boxes_areas = ((remaining_boxes[:, 2] - remaining_boxes[:, 0]) *
                              (remaining_boxes[:, 3] - remaining_boxes[:, 1]))

    # union = A + B - overlap (to avoid double counting intersection)
    union_areas = reference_box_area + remaining_boxes_areas - intersection_areas

    # guard against zero union — returns 0.0 for those boxes
    # np.where: if union > 0 → compute IoU, else → 0.0
    iou_scores = np.where(
        union_areas > 0,
        intersection_areas / union_areas,
        0.0
    )

    return iou_scores  # shape [N] — one IoU score per remaining box


# ─────────────────────────────────────────────
# FUNCTION 2 — calculate_IoU  )
# ─────────────────────────────────────────────
# Used by field_checker and other modules that compare two boxes one at a time
# Kept separate from vectorized version — different use cases

def calculate_IoU(boxA, boxB):

    #  min(boxA[3, boxB[3]) — comma INSIDE brackets = syntax error
    #  min(boxA[3], boxB[3]) — comma BETWEEN arguments
    left_edge_x  = max(boxA[0], boxB[0])
    topside_y    = max(boxA[1], boxB[1])
    right_edge_x = min(boxA[2], boxB[2])
    bottomside_y = min(boxA[3], boxB[3])

    interacted_intersection_width  = max(0.0, right_edge_x - left_edge_x)
    interacted_intersection_height = max(0.0, bottomside_y - topside_y)
    intersection_rectangle_area    = (interacted_intersection_width *
                                      interacted_intersection_height)

    if intersection_rectangle_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])

    # BUG WAS: (boxB[2]-boxB[0]) & (boxB[3]-boxB[1])
    # & is BITWISE AND — converts floats to ints then does binary operation
    # example: 120 & 80 = 16 (completely wrong area)
    # FIX: use * for multiplication
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    rectangles_union_area = boxA_area + boxB_area - intersection_rectangle_area

    if rectangles_union_area <= 0:
        return 0.0

    final_IoU_confidence_score = intersection_rectangle_area / rectangles_union_area
    return final_IoU_confidence_score


# ─────────────────────────────────────────────
# FUNCTION 3 — get_sigma_for_camera_angle
# ─────────────────────────────────────────────
# Research: sigma must adapt to camera angle — not fixed at 0.5
#
# WIDE SHOT (sigma 0.485):
# Players spread across field → overlapping boxes = likely duplicates
# More aggressive decay → kill duplicates faster
# Research: "wide shots have fewer true overlaps, duplicates more likely"
#
# TRENCH SHOT (sigma 0.84):
# Linemen physically touching → overlapping boxes = likely REAL players
# Gentle decay → keep neighboring real players alive
# Research: "larger sigma helps when high overlap tolerance is desired"
#
# UNKNOWN (sigma 0.5):
# Paper default — safe middle ground when we can't trust camera classification

def get_sigma_for_camera_angle(camera_angle_type):

    # wide — players in space, duplicates more likely than real neighbors
    if camera_angle_type == "wide":
        return 0.485

    # trench — linemen touching, real neighbors must survive
    elif camera_angle_type == "trench":
        return 0.84

    # unknown or any other value — use paper default
    else:
        return 0.5


# ─────────────────────────────────────────────
# FUNCTION 4 — validate_boxes
# ─────────────────────────────────────────────
# Research: "Soft-NMS behaves pathologically on invalid geometry"
# IoU math becomes meaningless with zero or negative area boxes
# Must run BEFORE the main loop — not inside it
#
# Catches 5 failure modes:
# 1. Wrong number of coordinates
# 2. Swapped x or y values (zero/negative area)
# 3. Non-finite values (NaN, Inf from corrupt frames)
# 4. Zero area (x1==x2 or y1==y2)
# 5. Returns original indices so tracking stays aligned

def validate_boxes(boxes, scores):

    valid_boxes  = []
    valid_scores = []
    valid_idxs   = []

    for i, box in enumerate(boxes):

        # check 1 — must have exactly 4 coordinates
        if len(box) != 4:
            print(f"  ⚠️  Box {i} skipped — wrong length {len(box)}")
            continue

        # BUG WAS: x1, x2, y1, y2 = box  ← WRONG ORDER
        # xyxy format is always: x1, y1, x2, y2
        # swapping x2 and y1 makes ALL geometry checks wrong
        # FIX: unpack in correct order
        x1, y1, x2, y2 = box

        # check 2 — must have positive area (strictly greater than)
        # x2 == x1 means zero-width box → union math breaks
        if x2 <= x1 or y2 <= y1:
            print(f"  ⚠️  Box {i} skipped — zero or negative area")
            continue

        # check 3 — all values must be real finite numbers
        # NaN or Inf from corrupt video frames causes silent wrong math
        # BUG WAS: mismatched brackets np.isfinite([x1,y1,x2,y2))
        # FIX: np.isfinite([x1, y1, x2, y2])
        # BUG WAS: extra closing paren on print ))
        # FIX: single closing paren
        if not all(np.isfinite([x1, y1, x2, y2])):
            print(f"  ⚠️  Box {i} skipped — non-finite coordinates")
            continue

        valid_boxes.append(box)
        valid_scores.append(scores[i])
        valid_idxs.append(i)  # track original position for Phase 5 tracker

    return valid_boxes, valid_scores, valid_idxs


# ─────────────────────────────────────────────
# FUNCTION 5 — gaussian_soft_nms  (MAIN)
# ─────────────────────────────────────────────
# Production-grade Gaussian Soft-NMS for football player detection
#
# TWO THRESHOLD SYSTEM (research: Detectron2 / MMCV pattern):
#   lower_pruning_threshold  = 1e-3  → speed only, cuts obvious noise MID-LOOP
#   higher_pruning_threshold = 0.25  → quality, defines "is this a player?" AT END
#
# Mixing them into one threshold is the most common Soft-NMS mistake
# A box decayed to 0.18 mid-loop might be a real occluded lineman
# Pruning at 0.25 kills it — we'd miss a real player
# Pruning at 1e-3 keeps it alive until final filter makes the real decision
#
# PIPELINE ROLE:
# After this runs → fused bbox_confidence scoring → ByteTrack tracker
# For counting/camera inference: higher_pruning_threshold = 0.25-0.50
# For tracker input: lower_pruning_threshold strictly, let tracker stabilize

def gaussian_soft_nms(
    boxes,
    scores,
    camera_angle_type        = "unknown",
    lower_pruning_threshold  = 1e-3,   # speed — prune obvious noise mid-loop
    higher_pruning_threshold = 0.25,   # quality — final "is this a player?" gate
    IoU_kill_level           = 0.95,   # hard kill — 95%+ overlap = true duplicate
):
    # ── guard: empty inputs ───────────────────────────────────────
    if not boxes or not scores:
        return [], []

    if len(boxes) != len(scores):
        print("ERROR: boxes and scores must be the same length")
        return [], []

    # ── validate geometry before any IoU math runs ────────────────
    # research: invalid boxes make IoU meaningless and corrupt output
    boxes, scores, original_idxs = validate_boxes(boxes, scores)

    if not boxes:
        print("  ⚠️  No valid boxes survived validation")
        return [], []

    # ── get adaptive sigma from camera angle ─────────────────────
    # research: fixed sigma=0.5 is wrong for trench play
    sigma = get_sigma_for_camera_angle(camera_angle_type)
    print(f"  Soft-NMS sigma: {sigma} (camera: {camera_angle_type})")

    # ── COPY PROTECTION — never mutate originals ──────────────────
    # research: "production implementations clone before operating"
    # film_analyzer.py needs original lists intact for draw_player_boxes
    # BUG WAS: directly called boxes.pop() on the input list
    # FIX: work on copies — originals stay untouched
    working_boxes_copy  = np.array(boxes, dtype=np.float32)
    working_scores_copy = np.array(scores, dtype=np.float32)
    working_idxs_copy   = np.array(original_idxs, dtype=np.int32)

    kept_bbox_list  = []
    kept_idx_list   = []

    # ── main Soft-NMS loop ────────────────────────────────────────
    while working_boxes_copy.shape[0] > 0:

        # step 1 — find highest scoring box in remaining list
        highest_indexed_bbox_confidence = int(
            np.argmax(working_scores_copy)
        )

        best_picked_bbox  = working_boxes_copy[highest_indexed_bbox_confidence]
        best_bbox_score   = float(working_scores_copy[highest_indexed_bbox_confidence])
        best_original_idx = int(working_idxs_copy[highest_indexed_bbox_confidence])

        # step 2 — add best box to keep list BEFORE removing it
        # BUG RISK: if you pop first, index shifts and you grab wrong box
        kept_bbox_list.append({
            "bbox":         best_picked_bbox.tolist(),
            "confidence":   best_bbox_score,
            "original_idx": best_original_idx   # tracker needs this
        })
        kept_idx_list.append(best_original_idx)

        # step 3 — remove best box from working copies
        # np.delete is safe on NumPy arrays — no index shifting bugs
        working_boxes_copy  = np.delete(working_boxes_copy,
                                        highest_indexed_bbox_confidence, axis=0)
        working_scores_copy = np.delete(working_scores_copy,
                                        highest_indexed_bbox_confidence)
        working_idxs_copy   = np.delete(working_idxs_copy,
                                        highest_indexed_bbox_confidence)

        if working_boxes_copy.shape[0] == 0:
            break

        # step 4 — VECTORIZED IoU: compute all overlaps at once
        # research: NumPy vectorized = 3-15x faster than Python loop
        # replaces: for i, box in enumerate(boxes): iou = calculate_IoU(...)
        all_iou_scores = compute_iou_vectorized(best_picked_bbox,
                                                working_boxes_copy)

        # step 5 — hard kill near-identical boxes
        # research: "engineering add-on for near-duplicate kill switch"
        # IoU >= 0.95 means two boxes describe the exact same player
        # no need to decay — just delete immediately
        not_duplicate_mask = all_iou_scores < IoU_kill_level

        working_boxes_copy  = working_boxes_copy[not_duplicate_mask]
        working_scores_copy = working_scores_copy[not_duplicate_mask]
        working_idxs_copy   = working_idxs_copy[not_duplicate_mask]
        all_iou_scores      = all_iou_scores[not_duplicate_mask]

        if working_boxes_copy.shape[0] == 0:
            break

        # step 6 — Gaussian decay formula (Bodla et al. 2017)
        # s_i ← s_i * exp( -(IoU² / sigma) )
        # high overlap → large exponent → heavy score decay
        # zero overlap → exp(0) = 1.0 → no decay at all
        # BUG WAS: deayed_score = scores[1] * ... (typo + hardcoded index)
        # FIX: vectorized across all remaining boxes at once
        gaussian_decay    = np.exp(-(all_iou_scores ** 2) / sigma)
        working_scores_copy = working_scores_copy * gaussian_decay

        # step 7 — prune for speed (lower_pruning_threshold)
        # research: "prune_thresh 1e-3 to 1e-2 — for speed only"
        # this is NOT the final player decision — just kills obvious noise
        # BUG WAS: boxes = surviving_boxes inside the for loop
        # FIX: vectorized mask applied after full decay pass
        # BUG WAS: used score_thresh (undefined variable)
        # FIX: use lower_pruning_threshold
        speed_prune_mask    = working_scores_copy >= lower_pruning_threshold
        working_boxes_copy  = working_boxes_copy[speed_prune_mask]
        working_scores_copy = working_scores_copy[speed_prune_mask]
        working_idxs_copy   = working_idxs_copy[speed_prune_mask]

    # ── final quality filter (higher_pruning_threshold) ───────────
    # research: "final_thresh is your real is-this-a-player decision"
    # applied AFTER all decay is complete — never mid-loop
    # BUG WAS: higher_pruning_threshold defined but never applied
    # FIX: filter kept_bbox_list here at the end
    final_kept = [
        detection for detection in kept_bbox_list
        if detection["confidence"] >= higher_pruning_threshold
    ]

    print(f"  Soft-NMS: {len(boxes)} boxes in "
          f"→ {len(final_kept)} players out "
          f"(sigma={sigma}, cam={camera_angle_type})")

    # BUG WAS: return inside while loop — returned after first iteration only
    # FIX: return is outside all loops — runs once everything is done
    return final_kept, kept_idx_list