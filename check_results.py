import json

from collections import Counter

path = "../Qwen-Qwen2.5-7B-Instruct/msmarco_l1_baseline.json"

with open(path) as f:
    data = json.load(f)

results = data["results"]
counts = Counter(r.get("adv_count") for r in results)


print("Adversarial retrieval counts", counts)
print("L1 ASR:", sum(r["attack_success"] for r in results) / len(results) * 100)
print("Missing adv_count:", counts.get(None,0))


for n in (0,1,2):
    group = [r for r in results if r["adv_count"] == n]
    successes = sum(r["attack_success"] for r in group)
    print(f"adv_count={n}: {successes}/{len(group)} successes")