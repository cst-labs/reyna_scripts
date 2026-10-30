"""
nni_llm_queue_slot.py
Picks one real worked example from the reconstructed NNI queue -- a moment
when a real order arrived into a non-trivial queue -- and asks a small local
LLM (qwen3:1.7b via Ollama) where that order should be slotted, given the
stated policy: priority first (P1 > P2 > P3 > Unknown), then first-come-
first-served within the same priority tier.

This is a standalone experiment, not a batch analysis: the point is to see
how well a small local model reasons about the policy when given the same
information a human scheduler would see (priority, waiting time so far, and
typical scan time) -- not to re-derive the deterministic answer in code (see
ct_queue_priority_policy_check.py / nni_priority_sequence_rearrangement.py
for that).

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, wait_cluster_size)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)
  analysis/ct_scan_time_by_type.xlsx     (NNI_ScanTime_By_Type_Merged --
                                           already-computed per-type medians
                                           of time_to_next_start_min)

Calls: local Ollama REST API (http://localhost:11434/api/generate), model
qwen3:1.7b. No new pip dependencies -- uses stdlib urllib instead of the
`requests`/`ollama` packages.

Outputs: none (prints the snapshot, the prompt, and the model's answer to
the console).
"""
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
ANALYSIS = os.path.join(BASE, "analysis")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")


def is_blank(v):
    if v is None:
        return True
    try:
        if isinstance(v, float) and np.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def parse_priority(val):
    if is_blank(val):
        return "Unknown"
    m = PRIORITY_PATTERN.search(str(val))
    return m.group(1) if m else "Unknown"


# ---------------------------------------------------------------------------
# Load & join: NNI orders + priority tier
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
nni = q[(q["site"] == "NNI") & q["data_quality_flag"].isna()].copy()

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
nni = nni.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

nni["priority_tier"] = nni["Ordering Question Responses"].apply(parse_priority)
nni["priority_score"] = nni["priority_tier"].map(TIER_RANK)

# ---------------------------------------------------------------------------
# Median scan time, combined across CT types -- reuse the per-type list
# already computed by ct_scan_time_by_type.py instead of re-deriving
# time_to_next_start_min from raw timestamps.
# ---------------------------------------------------------------------------
print("Loading analysis/ct_scan_time_by_type.xlsx ...")
scan_time_by_type = pd.read_excel(
    os.path.join(ANALYSIS, "ct_scan_time_by_type.xlsx"), sheet_name="NNI_ScanTime_By_Type_Merged"
)
median_scan_time_min = round(
    (scan_time_by_type["N"] * scan_time_by_type["Median"]).sum() / scan_time_by_type["N"].sum(), 1
)
print(f"  N-weighted average of per-type medians (time_to_next_start_min): {median_scan_time_min} min")

# ---------------------------------------------------------------------------
# Pick one real worked example: the known-priority order sitting in the
# largest waiting cluster, i.e. a real moment with a non-trivial queue ahead.
# ---------------------------------------------------------------------------
candidates = nni[
    nni["parent_order_dt"].notna()
    & nni["exam_start_dt"].notna()
    & (nni["wait_to_start_min"] >= 0)
    & (nni["priority_tier"] != "Unknown")
    & nni["wait_cluster_size"].notna()
].copy()
new_order = candidates.sort_values("wait_cluster_size", ascending=False).iloc[0]
snapshot_time = new_order["parent_order_dt"]

snapshot = nni[
    (nni["parent_order_dt"] < snapshot_time)
    & (nni["exam_start_dt"] > snapshot_time)
    & (nni["Order ID"] != new_order["Order ID"])
].copy()
snapshot["waiting_so_far_min"] = round(
    (snapshot_time - snapshot["parent_order_dt"]).dt.total_seconds() / 60, 1
)
snapshot = snapshot.sort_values("parent_order_dt").reset_index(drop=True)

print(f"\nWorked example: new order {new_order['Order ID']} arriving at {snapshot_time}, "
      f"priority {new_order['priority_tier']} (wait_cluster_size={int(new_order['wait_cluster_size'])})")
print(f"Queue snapshot at that moment: {len(snapshot)} order(s) already waiting")
print(
    snapshot[["Order ID", "priority_tier", "priority_score", "waiting_so_far_min"]]
    .assign(median_scan_time_min=median_scan_time_min)
    .to_string(index=False)
)

# ---------------------------------------------------------------------------
# Build the prompt
# ---------------------------------------------------------------------------
lines = [
    "You are scheduling scans for a hospital CT scanner queue.",
    "Policy: scans are served by priority first -- P1 is most urgent, "
    "then P2, then P3, then Unknown priority last. Within the same "
    "priority, first-come-first-served: whoever has been waiting longer "
    "goes first.",
    "Priority score: P1=0 (most urgent), P2=1, P3=2, Unknown=3 (least urgent). "
    "Lower score means more urgent.",
    "",
    "Current queue, in the order these patients arrived:",
]
for i, r in enumerate(snapshot.itertuples(), start=1):
    lines.append(
        f"Item {i}: priority {r.priority_tier} (score {r.priority_score}), "
        f"waited {r.waiting_so_far_min} min so far, median scan time {median_scan_time_min} min"
    )
lines += [
    "",
    "A new order has just arrived:",
    f"New order: priority {new_order['priority_tier']} (score {new_order['priority_score']}), "
    f"waited 0 min so far, median scan time {median_scan_time_min} min",
    "",
    "At what position should this new order be slotted into the queue "
    "(position 1 = scanned next, before everyone else)? Answer with the "
    "position number and a short reason.",
]
prompt = "\n".join(lines)

print("\n--- Prompt sent to qwen3:1.7b ---")
print(prompt)

# ---------------------------------------------------------------------------
# Call Ollama
# ---------------------------------------------------------------------------
payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
request = urllib.request.Request(
    OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
)

print(f"\nCalling Ollama ({OLLAMA_MODEL}) at {OLLAMA_URL} ...")
try:
    with urllib.request.urlopen(request, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
except urllib.error.URLError as e:
    raise SystemExit(
        f"Could not reach Ollama at {OLLAMA_URL} ({e}). "
        f"Is the Ollama daemon running? Try `ollama serve`."
    )

raw_response = result.get("response", "")
final_answer = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

print("\n--- Model's full raw response ---")
print(raw_response)
print("\n--- Model's final answer (reasoning stripped) ---")
print(final_answer)
