import json

import numpy as np


def decode(record):
    return {
        **record,
        "id": record["id"].decode("utf-8"),
        "vehicle": json.loads(record["vehicle"]),
        "track": np.frombuffer(record["track"], dtype=np.float32).reshape((-1, 4)),
        "widths": np.frombuffer(record["widths"], dtype=np.float32),
        "angles": np.frombuffer(record["angles"], dtype=np.float32),
        "offsets": np.frombuffer(record["offsets"], dtype=np.float32),
        "pos": np.frombuffer(record["pos"], dtype=np.float32),
        "acc": np.frombuffer(record["acc"], dtype=np.float32),
        "vel": np.frombuffer(record["vel"], dtype=np.float32),
        "flipped": bool(int.from_bytes(record["flipped"]))
    }


def encode(record):
    return {
        **record,
        "track": np.array(record["track"], dtype=np.float32).tobytes(),
        "vehicle": json.dumps(record["vehicle"]),
        "pos": np.array(record["pos"], dtype=np.float32).tobytes(),
        "vel": np.array(record["vel"], dtype=np.float32).tobytes(),
        "acc": np.array(record["acc"], dtype=np.float32).tobytes(),
        "widths": np.array(record["widths"], dtype=np.float32).tobytes(),
        "angles": np.array(record["angles"], dtype=np.float32).tobytes(),
        "offsets": np.array(record["offsets"], dtype=np.float32).tobytes(),
        "flipped": int(record["flipped"]).to_bytes(),
    }