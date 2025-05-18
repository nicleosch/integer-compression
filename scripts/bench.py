#!/usr/bin/env python3

import csv
import subprocess
import sys
from itertools import product
from pathlib import Path

output_file = "results.tbl"

# Define configurations
schemes = ["tinyblocks", "btrblocks"]
block_sizes = [64, 128, 256, 512]
depths = [1,2,3]

# Write header to output
with open(output_file, "w") as out:
    out.write("dataset|column|type|uncompressed_size|scheme|block_size|nullable|compressed_size|compressed_header|compressed_payload|compression_ratio\n")

# Parse output from the compression executable
def parse_output(output):
    result = {}
    for line in output.splitlines():
        parts = line.split(":")
        if len(parts) < 2:
            continue
        key = parts[0].strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        value = parts[1].strip().split(" ")[0]
        result[key] = value
    return result

def main():
    input_dir = Path(sys.argv[1])
    executable = sys.argv[2]

    if not input_dir.is_dir():
        print(f"[ERROR] {input_dir} is not a valid directory")
        sys.exit(1)

    data_files = list(input_dir.rglob("*.csv")) + list(input_dir.rglob("*.tbl"))
    if not data_files:
        print(f"[ERROR] No .csv or .tbl files found in {input_dir}")
        sys.exit(1)

    for file_path in data_files:
        with open(file_path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            try:
                headers = next(reader)
            except StopIteration:
                print(f"[WARN] Skipping empty file: {file_path}")
                continue

        for index, column in enumerate(headers):
            print(f"[→] Benchmarking {file_path}:{column}")
            for scheme, block_size, depth in product(schemes, block_sizes, depths):
                if scheme not in ["tinyblocks", "forn"] and block_size > 64:
                    continue  # block size only relevant for some schemes
                if scheme != "btrblocks" and depth > 1:
                    continue

                print(f"Executing {str(file_path)}, {index}, {scheme}, {block_size}")

                cmd = list(filter(None, [
                    executable,
                    "--data", str(file_path),
                    "--column", str(index),
                    "--type", "int",
                    "--delimiter", ",",
                    "--scheme", scheme,
                    "--size", str(block_size),
                    "--depth", str(depth)
                ]))

                try:
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                    stats = parse_output(output)

                    with open(output_file, "a") as out:
                        out.write(f"{file_path}|{column}|int|{stats.get('uncompressed_size', '?')}|{scheme if scheme == 'tinyblocks' else scheme+str(depth)}|{block_size}|false|{stats.get('compressed_size', '?')}|{stats.get('compressed_header_size', '?')}|{stats.get('compressed_payload_size', '?')}|{stats.get('compression_rate', '?')}\n")

                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Failed on {file_path}:{column}:{scheme} → {e.output.strip()}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 run_experiments.py <input_dir> <path_to_executable>")
        sys.exit(1)
    main()
