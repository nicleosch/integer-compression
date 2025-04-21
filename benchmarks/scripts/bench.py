#!/usr/bin/env python3

import csv
import subprocess
import sys
from itertools import product

output_file = "results.tbl"

# Define configurations
schemes = ["tinyblocks", "bitpacking", "delta", "for", "rle", "forn", "lz4", "zstd", "snappy", "btrblocks", "uncompressed"]
block_sizes = [64, 128, 256, 512]
p2schemes = ["lz4", "zstd", "snappy"]
non_morsel_schemes = set(["lz4", "zstd", "snappy", "btrblocks", "rle", "delta"])
non_datablock_schemes = non_morsel_schemes | set(["uncompressed"])

# Initialize output file with header
with open(output_file, "w") as out:
    out.write("dataset|column|type|uncompressed_size|scheme|block_size|morsel|p2schema|p2header|p2payload|compressed_size|compressed_header|compressed_payload|compression_rate|decompression_time|decompression_bandwidth(compressed)|decompression_bandwidth(uncompressed)\n")

# Helper to parse the output from the executable
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
    # Process each row from the input file
    input_file = sys.argv[1]
    executable = sys.argv[2]
    data_prefix = sys.argv[3].rstrip("/") + "/"
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            dataset_path = row[0]
            column = row[1]
            col_type = row[2]

            print(f"Starting benchmark on {dataset_path}:{column}.")

            for scheme, block_size in product(schemes, block_sizes):
                if scheme not in ["tinyblocks", "forn"] and block_size > 64:
                    # block size only relevant for tinyblocks and forn
                    continue

                # Test with morsel
                cmd = list(filter(None, [
                    executable,
                    "--data", data_prefix + dataset_path,
                    "--column", column,
                    "--type", "int" if col_type == "integer" else col_type,
                    "--delimiter", "|",
                    "--scheme", scheme,
                    "--size", str(block_size),
                    "--morsel" if scheme not in non_morsel_schemes else None
                ]))

                try:
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                    stats = parse_output(output)

                    with open(output_file, "a") as out:
                        out.write(f"{dataset_path}|{column}|{col_type}|{stats['uncompressed_size']}|{scheme}|{block_size}|true||||{stats['compressed_size']}|{stats['compressed_header_size']}|{stats['compressed_payload_size']}|{stats['compression_rate']}|{stats['decompression_time']}|{stats['decompression_bandwidth_compressed']}|{stats['decompression_bandwidth_uncompressed']}\n")

                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Morsel cmd failed for {dataset_path}:{column}:{scheme} | {e.output}")
                    print(e)


                if scheme in non_datablock_schemes:
                    continue
                # Test without morsel + p2scheme combos
                for p2scheme, p2header, p2payload in product(p2schemes, [True, False], [True, False]):
                    cmd = [
                        executable,
                        "--data", data_prefix + dataset_path,
                        "--column", column,
                        "--type", "int" if col_type == "integer" else col_type,
                        "--delimiter", "|",
                        "--scheme", scheme,
                        "--size", str(block_size),
                        "--p2scheme", p2scheme
                    ]
                    if p2header:
                        cmd.append("--p2header")
                    if p2payload:
                        cmd.append("--p2payload")

                    try:
                        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                        stats = parse_output(output)

                        with open(output_file, "a") as out:
                            out.write(f"{dataset_path}|{column}|{col_type}|{stats['uncompressed_size']}|{scheme}|{block_size}|false|{p2scheme}|{str(p2header).lower()}|{str(p2payload).lower()}|{stats['compressed_size']}|{stats['compressed_header_size']}|{stats['compressed_payload_size']}|{stats['compression_rate']}|{stats['decompression_time']}|{stats['decompression_bandwidth_compressed']}|{stats['decompression_bandwidth_uncompressed']}\n")

                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] Non-morsel cmd failed for {dataset_path}:{column}:{scheme} | {e.output}")
                        print(e)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 run_experiments.py <input_file> <path_to_executable> <data_prefix>")
        sys.exit(1)
    main()