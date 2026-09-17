"""
intensity_scorer.py

Intensity score in [0,1] for a dynamic gesture recording.
Measures how quickly/explosively the tracked finger configuration changed.
The recording does not contain wrist translation.

Data format: JSON list of {confidence, sequenceData: [[f0..f16] x T]}
Feature layout (17 floats per frame):
  [0,1]       thumb curl, abduction
  [2,3,4,5]   index curl, abduction, flexion, opposition
  [6,7,8,9]   middle curl, abduction, flexion, opposition
  [10,11,12,13] ring curl, abduction, flexion, opposition
  [14,15,16]  pinky curl, flexion, opposition

Curl, abduction, and flexion are angle features measured in degrees.
Opposition is fingertip-to-thumb distance measured in metres.
The two unit groups are measured separately before normalization.
"""

import json
import numpy as np

CURL_IDX = [0, 2, 6, 10, 14]
ABDUCTION_IDX = [1, 3, 7, 11]
FLEXION_IDX = [4, 8, 12, 15]
OPPOSITION_IDX = [5, 9, 13, 16]
ANGLE_IDX = CURL_IDX + ABDUCTION_IDX + FLEXION_IDX
FRAME_RATE = 30
DT = 1.0 / FRAME_RATE

# PCA on the corrected features gave 0.3545, 0.3234, and 0.3221.
# These rounded weights keep the three intensity measurements nearly balanced.
W = {"angular_speed": 0.36, "angular_accel": 0.32, "opposition_speed": 0.32}
assert abs(sum(W.values()) - 1.0) < 1e-9


def load_recording(path):
    with open(path) as f:
        raw = json.load(f)
    return [
        (np.array(e["sequenceData"], dtype=np.float32), e["confidence"])
        for e in raw
    ]


def extract_features(frames):
    # RMS keeps a group from appearing faster solely because it has more features.
    angle_deltas = np.diff(frames[:, ANGLE_IDX], axis=0)  # degrees
    angular_speed = np.sqrt(np.mean(angle_deltas ** 2, axis=1)) / DT

    opposition_deltas = np.diff(frames[:, OPPOSITION_IDX], axis=0)  # metres
    opposition_speed = np.sqrt(np.mean(opposition_deltas ** 2, axis=1)) / DT

    avg_angular_speed = float(np.mean(angular_speed))
    avg_opposition_speed = float(np.mean(opposition_speed))
    avg_angular_accel = (
        float(np.mean(np.abs(np.diff(angular_speed)) / DT))
        if len(angular_speed) > 1
        else 0.0
    )

    return avg_angular_speed, avg_angular_accel, avg_opposition_speed


def calibrate(all_seqs):
    feats = [extract_features(s) for s in all_seqs]
    feats = np.array(feats)  # (N, 3): angular speed, angular acceleration, opposition speed
    return feats.min(axis=0), feats.max(axis=0)


def score(frames, feat_min, feat_max, confidence):
    raw = np.array(extract_features(frames))
    normed = np.clip((raw - feat_min) / (feat_max - feat_min + 1e-8), 0, 1)
    angular_speed_s, angular_accel_s, opposition_speed_s = normed
    intensity = (
        W["angular_speed"] * angular_speed_s
        + W["angular_accel"] * angular_accel_s
        + W["opposition_speed"] * opposition_speed_s
    )
    return {
        "intensity": round(float(intensity), 4),
        # Previously speed_score, acceleration_score, and curl_speed_score.
        # The new names reflect the corrected angle and opposition feature groups.
        "angular_speed_score": round(float(angular_speed_s), 4),
        "angular_acceleration_score": round(float(angular_accel_s), 4),
        "opposition_speed_score": round(float(opposition_speed_s), 4),
        "confidence": confidence,  # Placeholder for a future learned confidence value.
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
    print(f"Calibration — angular speed: [{feat_min[0]:.1f}, {feat_max[0]:.1f}]  "
          f"angular acceleration: [{feat_min[1]:.1f}, {feat_max[1]:.1f}]  "
          f"opposition speed: [{feat_min[2]:.4f}, {feat_max[2]:.4f}]\n")

    for name, entries in all_data.items():
        intensities = []
        print(f"=== {name} ===")
        for i, (frames, conf) in enumerate(entries):
            result = score(frames, feat_min, feat_max, conf)
            intensities.append(result["intensity"])
            print(f"  [{i+1:2d}] intensity={result['intensity']:.3f}  "
                  f"angular_speed={result['angular_speed_score']:.3f}  "
                  f"angular_accel={result['angular_acceleration_score']:.3f}  "
                  f"opposition_speed={result['opposition_speed_score']:.3f}")
        print(f"  avg={np.mean(intensities):.3f}  "
              f"min={min(intensities):.3f}  max={max(intensities):.3f}\n")
