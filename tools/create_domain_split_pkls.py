import os
import pickle
from collections import Counter

from nuscenes.nuscenes import NuScenes

VAL_PKL_PATH = r"C:\datasets\nuscenes\nuscenes_infos_temporal_val.pkl"
DATA_ROOT = r"C:\datasets\nuscenes"
OUT_BOSTON = os.path.join(DATA_ROOT, "nuscenes_infos_temporal_val_boston.pkl")
OUT_SINGAPORE = os.path.join(DATA_ROOT, "nuscenes_infos_temporal_val_singapore.pkl")


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _build_scene_location_map(nusc):
    scene_to_location = {}
    for scene in nusc.scene:
        log = nusc.get("log", scene["log_token"])
        scene_to_location[scene["token"]] = log["location"].lower()
    return scene_to_location


def _split_infos(infos, scene_to_location):
    boston_infos = []
    singapore_infos = []
    unknown_infos = []

    for info in infos:
        location = None
        scene_token = info.get("scene_token")
        if scene_token:
            location = scene_to_location.get(scene_token)

        # Fallback for non-standard infos
        if not location:
            lidar_path = str(info.get("lidar_path", "")).lower()
            if "boston" in lidar_path:
                location = "boston"
            elif "singapore" in lidar_path:
                location = "singapore"

        if location and "boston" in location:
            boston_infos.append(info)
        elif location and "singapore" in location:
            singapore_infos.append(info)
        else:
            unknown_infos.append(info)

    return boston_infos, singapore_infos, unknown_infos


def main():
    print(f"Loading PKL: {VAL_PKL_PATH}")
    data = _load_pkl(VAL_PKL_PATH)

    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
        metadata = data.get("metadata", {})
        out_kind = "legacy"
    elif isinstance(data, dict) and "data_list" in data:
        infos = data["data_list"]
        metadata = data.get("metainfo", {})
        out_kind = "new"
    else:
        raise ValueError("Unsupported PKL format: expected dict with infos/data_list")

    print(f"Total val samples: {len(infos)}")

    nusc = NuScenes(version="v1.0-trainval", dataroot=DATA_ROOT, verbose=False)
    scene_to_location = _build_scene_location_map(nusc)

    boston_infos, singapore_infos, unknown_infos = _split_infos(infos, scene_to_location)

    print(f"Boston val samples: {len(boston_infos)}")
    print(f"Singapore val samples: {len(singapore_infos)}")
    print(f"Unknown samples: {len(unknown_infos)}")

    if unknown_infos:
        print("WARNING: some samples could not be mapped to Boston/Singapore")

    if out_kind == "legacy":
        boston_payload = {"infos": boston_infos, "metadata": metadata}
        singapore_payload = {"infos": singapore_infos, "metadata": metadata}
    else:
        boston_payload = {"data_list": boston_infos, "metainfo": metadata}
        singapore_payload = {"data_list": singapore_infos, "metainfo": metadata}

    _save_pkl(OUT_BOSTON, boston_payload)
    _save_pkl(OUT_SINGAPORE, singapore_payload)

    print(f"Saved: {OUT_BOSTON}")
    print(f"Saved: {OUT_SINGAPORE}")


if __name__ == "__main__":
    main()
