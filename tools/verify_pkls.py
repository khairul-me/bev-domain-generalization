import os
import pickle

import numpy as np

TRAIN_PKL = r"C:\datasets\nuscenes\nuscenes_infos_temporal_train.pkl"
VAL_PKL = r"C:\datasets\nuscenes\nuscenes_infos_temporal_val.pkl"


def check_pkl(path, name):
    print(f"\n{'=' * 60}")
    print(f"Checking: {name}")
    print(f"Path: {path}")

    if not os.path.exists(path):
        print("ERROR: File does not exist.")
        return False

    size_mb = os.path.getsize(path) / 1e6
    print(f"File size: {size_mb:.1f} MB")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load pkl — {e}")
        return False

    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
    elif isinstance(data, list):
        infos = data
    else:
        if isinstance(data, dict):
            desc = data.keys()
        else:
            desc = type(data)
        print(f"ERROR: Unexpected pkl structure — keys: {desc}")
        return False

    print(f"Sample count: {len(infos)}")

    expected = 28130 if name == "TRAIN" else 6019
    if len(infos) < expected * 0.95:
        print(
            f"WARNING: Sample count {len(infos)} is significantly below expected {expected}."
        )
        print("         PKL may be from a partial/truncated run.")
    else:
        print(f"Sample count OK (expected ~{expected})")

    sample = infos[0]
    if "can_bus" not in sample:
        print("WARNING: 'can_bus' field is MISSING from samples entirely.")
        return False

    can_bus = np.array(sample["can_bus"])
    print(f"CAN bus field present: shape={can_bus.shape}")

    if np.all(can_bus == 0):
        print("WARNING: CAN bus is ALL ZEROS — zero-vector fallback is in effect.")
        print("         This PKL was generated with the broken fallback.")
        print("         ACTION REQUIRED: Regenerate PKLs after fixing CAN bus.")
        return "zeroed"

    print(f"CAN bus appears REAL — sample values: {can_bus[:4]}")
    print("CAN bus: OK")
    return True


train_result = check_pkl(TRAIN_PKL, "TRAIN")
val_result = check_pkl(VAL_PKL, "VAL")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"Train PKL: {train_result}")
print(f"Val PKL:   {val_result}")

if train_result is True and val_result is True:
    print("\nRESULT: PKLs are VALID. You do NOT need to regenerate them.")
    print("        Proceed directly to Phase 6.")
elif train_result == "zeroed" or val_result == "zeroed":
    print("\nRESULT: PKLs exist but contain ZEROED CAN bus data.")
    print("        Complete Phases 3-5 to regenerate with real CAN bus.")
else:
    print("\nRESULT: PKLs are MISSING or BROKEN.")
    print("        Complete Phases 3-5 to regenerate.")
