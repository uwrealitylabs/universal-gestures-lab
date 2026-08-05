"""
intensity_scorer.py

Intensity score in [0,1] for a dynamic gesture recording.
Measures how quickly/explosively the hand configuration changed —
NOT wrist translation, which is gesture-specific and unreliable.

Data format: JSON list of {confidence, sequenceData: [[f0..f16] x T]}
Feature layout (17 floats per frame):
  [0,1]       wrist x,y          (large ~175-230, gesture-dependent, IGNORED)
  [2,3,4]     index base x,y,tip-x
  [5]         index curl         (small ~0.01-0.12)
  [6,7,8]     middle x,y,z
  [9]         middle curl
  [10,11,12]  ring x,y,z
  [13]        ring curl
  [14,15,16]  pinky x,y,z + curl

We care about:
  - curl change rate  (fingers moving fast = intense)
  - finger joint change rate  (all non-wrist position deltas)
Both computed as frame-to-frame L2 deltas, then summarized as
avg velocity and avg acceleration of the hand *configuration*.
"""

import json
import numpy as np

CURL_IDX  = [5, 9, 13, 16]
JOINT_IDX = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # non-wrist positions
FRAME_RATE = 30
DT = 1.0 / FRAME_RATE

# Weights — acceleration weighted most, it best captures explosiveness
W = {"accel": 0.45, "speed": 0.35, "curl_speed": 0.20}
assert abs(sum(W.values()) - 1.0) < 1e-9


def load_recording(path):
    with open(path) as f:
        raw = json.load(f)
    return [
        (np.array(e["sequenceData"], dtype=np.float32), e["confidence"])
        for e in raw
    ]


def extract_features(frames):
    # Frame-to-frame delta of full hand configuration (joints only, no wrist)
    joint_deltas = np.diff(frames[:, JOINT_IDX], axis=0)   # (T-1, 12)
    joint_speed  = np.linalg.norm(joint_deltas, axis=1) / DT  # (T-1,)

    curl_deltas = np.diff(frames[:, CURL_IDX], axis=0)     # (T-1, 4)
    curl_speed  = np.linalg.norm(curl_deltas, axis=1) / DT    # (T-1,)

    avg_speed      = float(np.mean(joint_speed))
    avg_curl_speed = float(np.mean(curl_speed))
    avg_accel      = float(np.mean(np.abs(np.diff(joint_speed)))) if len(joint_speed) > 1 else 0.0

    return avg_speed, avg_accel, avg_curl_speed


def calibrate(all_seqs):
    feats = [extract_features(s) for s in all_seqs]
    feats = np.array(feats)  # (N, 3): speed, accel, curl_speed
    return feats.min(axis=0), feats.max(axis=0)


def score(frames, feat_min, feat_max, confidence):
    raw = np.array(extract_features(frames))
    normed = np.clip((raw - feat_min) / (feat_max - feat_min + 1e-8), 0, 1)
    speed_s, accel_s, curl_s = normed
    intensity = W["speed"] * speed_s + W["accel"] * accel_s + W["curl_speed"] * curl_s
    return {
        "intensity":          round(float(intensity), 4),
        "speed_score":        round(float(speed_s), 4),
        "acceleration_score": round(float(accel_s), 4),
        "curl_speed_score":   round(float(curl_s), 4),
        "confidence":         confidence, #confidence is just a placeholder right now, maybe add a learned model layer if you have time?
    }


if __name__ == "__main__":
    FILES = {
        "fire_finger_gun_pos1": "/mnt/user-data/uploads/dynamic_fire_finger_gun_pos1_2025-08-22_16-41-08.json",
        "fire_finger_gun_pos2": "/mnt/user-data/uploads/dynamic_fire_finger_gun_pos2_2025-08-22_18-19-24.json",
        "fire_finger_gun_pos3": "/mnt/user-data/uploads/dynamic_fire_finger_gun_pos3_2025-08-22_18-21-21.json",
        "fire_finger_gun_pos4": "/mnt/user-data/uploads/dynamic_fire_finger_gun_pos4_2025-08-26_16-37-04.json",
    }

    all_data = {name: load_recording(path) for name, path in FILES.items()}

    # Calibrate across everything we have — swap in negative examples later
    # as the floor once src/data(Dynamic)/no_gesture recordings are available
    all_seqs = [seq for entries in all_data.values() for seq, _ in entries]
    feat_min, feat_max = calibrate(all_seqs)
    print(f"Calibration — speed: [{feat_min[0]:.1f}, {feat_max[0]:.1f}]  "
          f"accel: [{feat_min[1]:.1f}, {feat_max[1]:.1f}]  "
          f"curl_speed: [{feat_min[2]:.4f}, {feat_max[2]:.4f}]\n")

    for name, entries in all_data.items():
        intensities = []
        print(f"=== {name} ===")
        for i, (frames, conf) in enumerate(entries):
            result = score(frames, feat_min, feat_max, conf)
            intensities.append(result["intensity"])
            print(f"  [{i+1:2d}] intensity={result['intensity']:.3f}  "
                  f"speed={result['speed_score']:.3f}  "
                  f"accel={result['acceleration_score']:.3f}  "
                  f"curl={result['curl_speed_score']:.3f}")
        print(f"  avg={np.mean(intensities):.3f}  "
              f"min={min(intensities):.3f}  max={max(intensities):.3f}\n")