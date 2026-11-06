#!/usr/bin/env python3

import subprocess
from pathlib import Path
import os

CONFIG_FILE = Path("config.yaml")
BASELINE_FILE = Path("baseline_configs.txt")

# ---------------------------------------------------------------------
# Read the original config.yaml
# ---------------------------------------------------------------------
with open(CONFIG_FILE, "r") as f:
    original = f.readlines()

# Find where the attack section begins
attack_start = None
for i, line in enumerate(original):
    if line.strip() == "attack:":
        attack_start = i
        break

if attack_start is None:
    raise RuntimeError("Could not locate 'attack:' section in config.yaml")

config_prefix = original[:attack_start]

# ---------------------------------------------------------------------
# Parse baseline_configs.txt
# ---------------------------------------------------------------------
blocks = []
current = []

with open(BASELINE_FILE) as f:
    for line in f:
        # Remove newline and convert any tabs to spaces
        line = line.rstrip("\n").expandtabs(2)

        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)

if current:
    blocks.append(current)

# ---------------------------------------------------------------------
# Run each attack configuration
# ---------------------------------------------------------------------
first = True

for block in blocks:

    output_name = block[0].strip()
    attack_yaml = "\n".join(block[1:]).expandtabs(2)

    with open(CONFIG_FILE, "w") as f:
        f.writelines(config_prefix)

        # Ensure exactly one newline between sections
        if not config_prefix[-1].endswith("\n"):
            f.write("\n")

        f.write(attack_yaml)
        f.write("\n")

    output_file = (
        f"/tmp/$USER/InceptionRAG_data/results/baselines/{output_name}.json"
    )

    cmd = [
        "python",
        "run_attack.py",
        "--config",
        str(CONFIG_FILE),
        "--dataset",
        "nq",
        "--limit",
        "100",
        "--data-root",
        f"/tmp/{os.environ['USER']}/InceptionRAG_data",
        "--output",
        output_file,
        "--add-gt",
        "--llm-judge",
    ]

    if first:
        cmd.append("--stage1")
        first = False

    print("=" * 80)
    print("Running:", output_name)
    print(" ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)

print("\nFinished all baseline experiments.")
