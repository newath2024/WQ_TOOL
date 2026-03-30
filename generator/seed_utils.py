from __future__ import annotations

import hashlib


def derive_generation_seed(
    base_seed: int,
    *,
    run_id: str,
    round_index: int,
    scope: str,
) -> int:
    payload = f"{int(base_seed)}:{run_id}:{int(round_index)}:{scope}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
