#!/usr/bin/env python3
"""
OMR-NED Evaluation Pipeline

Runs the full evaluation pipeline:
1. Start virtual display (xvnc)
2. Set DISPLAY=:0 for GUI tools (MuseScore)
3. Convert ABC predictions to MusicXML via utils/convert.py
4. Compute OMR-NED via scripts/compute_OMR-NED.py

Usage:
    PYTHONPATH=. python scripts/eval_pipeline_OMR-NED.py \
        --input_file path/to/predictions_abc.json \
        --ground_truth path/to/dataset
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="OMR-NED evaluation pipeline: xvnc -> convert ABC to XML -> compute OMR-NED"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file. Use xxx_abc.json for ABC predictions (converted to XML); or xxx_xml.json to skip conversion.",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to the ground truth dataset (HuggingFace dataset on disk) or JSON file",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="/dev/shm/tmp",
        help="Temporary directory for intermediate files (default: /dev/shm/tmp)",
    )
    parser.add_argument(
        "--skip_xvnc",
        action="store_true",
        help="Skip xvnc startup (use if DISPLAY is already set). The xvnc script path can be overridden with $XVNC_SCRIPT.",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    # Step 1: Start xvnc (virtual display) in background
    if not args.skip_xvnc:
        xvnc_path = os.environ.get("XVNC_SCRIPT", "/gscratch/ark/sharedmusic/software/xvnc.sh")
        if not os.path.isfile(xvnc_path):
            print(f"[WARNING] xvnc script not found at {xvnc_path}, skipping...")
        else:
            print("Step 1: Starting xvnc in background...")
            subprocess.Popen(
                ["bash", xvnc_path],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)  # Allow Xvnc to initialize before using DISPLAY
            print("xvnc started.")

    # Step 2: Set DISPLAY
    os.environ["DISPLAY"] = ":0"
    print("Step 2: DISPLAY=:0")

    # Step 3: Convert ABC to XML (if input is ABC)
    if args.input_file.endswith("_abc.json") or "_abc." in os.path.basename(args.input_file):
        print("Step 3: Converting ABC to MusicXML...")
        convert_cmd = [
            sys.executable,
            "utils/convert.py",
            "--input_file", args.input_file,
            "--tmp_dir", args.tmp_dir,
        ]
        result = subprocess.run(convert_cmd, env=os.environ.copy())
        if result.returncode != 0:
            print("[ERROR] ABC to XML conversion failed.")
            sys.exit(1)

        # Infer output XML path (convert.py replaces _abc.json with _xml.json)
        xml_file = args.input_file.replace("_abc.json", "_xml.json")
        if not os.path.isfile(xml_file):
            print(f"[ERROR] Expected XML output not found: {xml_file}")
            sys.exit(1)
        print(f"XML output: {xml_file}")
    else:
        # Input is already XML
        xml_file = args.input_file
        if not os.path.isfile(xml_file):
            print(f"[ERROR] XML file not found: {xml_file}")
            sys.exit(1)
        print("Step 3: Skipping conversion (input is already XML)")

    # Step 4: Compute OMR-NED
    print("Step 4: Computing OMR-NED...")
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    result = subprocess.run(
        [sys.executable, "scripts/compute_OMR-NED.py", "--prediction_file", xml_file, "--ground_truth", args.ground_truth],
        env=env,
    )
    if result.returncode != 0:
        print("[ERROR] OMR-NED computation failed.")
        sys.exit(1)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
