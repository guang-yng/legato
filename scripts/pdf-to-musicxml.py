import os
import subprocess
import glob
import json
import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pdf2image import convert_from_path


def pdf_to_png(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"[{time.ctime()}] Converting {pdf_path} to PNG pages...", flush=True)
    images = convert_from_path(pdf_path, dpi=300)
    for i, image in enumerate(images):
        page_path = os.path.join(output_folder, f"page_{i}.png")
        image.save(page_path, "PNG")
    print(
        f"[{time.ctime()}] Successfully created {len(images)} pages in '{output_folder}' directory.",
        flush=True,
    )


def run_batch_inference(image_folder, batch_size):
    print(f"\n[{time.ctime()}] --- Running Batch Inference (Step 2) ---", flush=True)
    print(
        f"[{time.ctime()}] Processing all images in '{image_folder}' with batch size {batch_size}...",
        flush=True,
    )
    env = {**os.environ, "PYTHONPATH": "."}
    inference_cmd = [
        sys.executable,
        "scripts/inference.py",
        "--model_path",
        "guangyangmusic/legato",
        "--image_path",
        image_folder,
        "--batch_size",
        str(batch_size),
        "--output_path",
        image_folder,
        "--fp16",
    ]
    try:
        print(
            f"[{time.ctime()}] Starting inference subprocess. This may take a long time...",
            flush=True,
        )
        subprocess.run(inference_cmd, check=True, env=env)
        print(f"[{time.ctime()}] Batch inference complete.", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[{time.ctime()}] ERROR: Batch inference failed with exit code {e.returncode}",
            flush=True,
        )
        return False


def run_abc_to_xml_conversion(output_folder, musescore_path):
    print(f"\n[{time.ctime()}] --- Converting ABC to MusicXML (Step 3) ---", flush=True)
    if not os.path.exists(musescore_path):
        print(
            f"[{time.ctime()}] FATAL ERROR: MuseScore not found at '{musescore_path}'",
            flush=True,
        )
        return False
    abc_json_files = sorted(glob.glob(os.path.join(output_folder, "*_abc.json")))
    if not abc_json_files:
        print(
            f"[{time.ctime()}] ERROR: No ABC JSON files found after inference step.",
            flush=True,
        )
        return False
    convert_script_path = "utils/convert.py"
    with open(convert_script_path, "r") as f:
        original_content = f.read()
    modified_content = original_content.replace(
        'XMLFIX_CMD = ["software/mscore"]', f'XMLFIX_CMD = ["{musescore_path}"]'
    )
    with open(convert_script_path, "w") as f:
        f.write(modified_content)
    try:
        for abc_path in abc_json_files:
            print(
                f"  [{time.ctime()}] Converting {os.path.basename(abc_path)}...",
                flush=True,
            )
            convert_cmd = [
                "xvfb-run",
                "-a",
                sys.executable,
                convert_script_path,
                "--input_file",
                abc_path,
            ]
            subprocess.run(convert_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"[{time.ctime()}] ERROR: ABC to XML conversion failed on {abc_path} with exit code {e.returncode}",
            flush=True,
        )
        return False
    finally:
        with open(convert_script_path, "w") as f:
            f.write(original_content)
    print(f"[{time.ctime()}] All conversions complete.", flush=True)
    return True


def stitch_musicxml_files(output_folder, final_xml_path):
    print(f"\n[{time.ctime()}] --- Stitching MusicXML Files (Step 4) ---", flush=True)
    xml_json_files = sorted(glob.glob(os.path.join(output_folder, "*_xml.json")))
    if not xml_json_files:
        print("No XML files to stitch.", flush=True)
        return
    with open(xml_json_files[0], "r") as f:
        xml_string = json.load(f)[0]
    base_tree = ET.fromstring(xml_string)
    base_parts = {part.get("id"): part for part in base_tree.findall("part")}
    all_measures = base_tree.findall(".//measure")
    measure_counter = int(all_measures[-1].get("number")) if all_measures else 0
    for file_path in xml_json_files[1:]:
        print(
            f"  [{time.ctime()}] Appending measures from {os.path.basename(file_path)}...",
            flush=True,
        )
        with open(file_path, "r") as f:
            xml_string = json.load(f)[0]
        next_tree = ET.fromstring(xml_string)
        for part in next_tree.findall("part"):
            part_id = part.get("id")
            if part_id in base_parts:
                for measure in part.findall("measure"):
                    measure_counter += 1
                    measure.set("number", str(measure_counter))
                base_parts[part_id].extend(part.findall("measure"))
    final_tree = ET.ElementTree(base_tree)
    final_tree.write(
        final_xml_path, encoding="utf-8", xml_declaration=True, method="xml"
    )
    print(
        f"\n[{time.ctime()}] Successfully stitched all pages into: {final_xml_path}",
        flush=True,
    )


def main(args):
    if not os.path.exists(args.pdf_path):
        print(
            f"[{time.ctime()}] ERROR: The input file was not found: {args.pdf_path}",
            flush=True,
        )
        return
    pdf_basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
    final_output_path = os.path.join(args.output_folder, f"{pdf_basename}.xml")
    pdf_to_png(args.pdf_path, args.output_folder)
    # we only support batch_size 1 with fp16 for now
    success_inference = run_batch_inference(args.output_folder, 1)
    if success_inference:
        success_conversion = run_abc_to_xml_conversion(
            args.output_folder, args.musescore_path
        )
        if success_conversion:
            stitch_musicxml_files(args.output_folder, final_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end PDF to MusicXML conversion pipeline for LEGATO."
    )
    parser.add_argument(
        "--pdf_path", type=str, required=True, help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to store intermediate PNG pages and other files.",
    )

    parser.add_argument(
        "--musescore_path",
        type=str,
        required=True,
        help="Path to the MuseScore executable.",
    )

    args = parser.parse_args()
    print(f"[{time.ctime()}] Starting pipeline with args: {args}", flush=True)
    main(args)
    print(f"[{time.ctime()}] Pipeline finished.", flush=True)
