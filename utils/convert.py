import os
import re
import json
import subprocess
from argparse import ArgumentParser
from tqdm import tqdm

ABC2XML_CMD = ["python", "utils/abc2xml.py"]
XMLFIX_CMD = ["software/mscore"]

def count_measure(abc_voice):
    return len(re.findall(r"\|\||\||::", abc_voice)) - (abc_voice.startswith('|'))

def generate_dummy_voice(num_measures, num_8th, ending):
    rest = f"x{num_8th}"
    cur_measures = 0
    content = ""
    while cur_measures + 5 < num_measures:
        content += f" {rest} | {rest} | {rest} | {rest} | {rest} | %{cur_measures + 5}\n"
        cur_measures += 5
    while cur_measures < num_measures:
        content += f" {rest} |"
        cur_measures += 1
    content += ending
    return content

def complete_brackets(s):
    pair = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for char in s:
        if char in pair:
            stack.append(char)
        elif char in pair.values():
            if stack and pair[stack[-1]] == char:
                stack.pop()
    for opener in reversed(stack):
        s += pair[opener]
    return s

def cleanup_abc(abc):
    abc = abc.replace("<|text|>", "text")

    # Fix voice name for piano
    m = re.search(r"V:1 (\w+)(.*)\n", abc)
    if m is None:
        print("No voice found. Use empty XML instead.")
        return ''
    abc = abc.replace(m.group(0), f"V:1 {m.group(1)} nm=\"Piano\" snm=\"Pno.\"\n")

    # Complete brackets in %%score
    match = re.search(r"(%%score )([^\n]*)\n", abc)
    if match:
        prefix, body = match.groups()
        fixed_body = complete_brackets(body)
        abc = abc.replace(match.group(0), prefix + fixed_body + "\n")

    # Remove strings info and lyrics
    abc = re.sub(r"strings=\"[^\"]*\"", "", abc)
    abc = re.sub(r"w: [^\n]*\n", "", abc) 

    m = [(match.end(), match.group(0)) for match in re.finditer(r"%\d+", abc)]
    if len(m) > 0:

        if len(abc) == m[-1][0]:
            abc = abc + "\n "

        elif len(abc) > m[-1][0] + 1:
            # Fill unfinished lines
            abc = abc[:m[-1][0] + 1]
            total_voices = max([int(v[2:]) for v in re.findall(r"V:\d+", abc)])
            voice_start_loc = re.search(r"(V:\d+\n([^V]+|$))+", abc).start()
            header = abc[:voice_start_loc]
            voice_contents = re.findall(r"V:\d+\n([^V]+|$)", abc[voice_start_loc:])
            assert len(voice_contents) > 0 
            metric = re.findall(r"M:(\d+/\d+)", abc)
            metric = "4/4" if len(metric) == 0 else metric[0]
            num_8th = 8 // int(metric.split("/")[1]) * int(metric.split("/")[0])
            if len(voice_contents) == 1:
                # Only one voice, truncate to the first repeated line
                lines = voice_contents[0].split("\n")
                ending_line = len(lines)-1
                for i in range(len(lines)-1):
                    if lines[i].split('%')[0] == lines[i+1].split('%')[0]:
                        ending_line = i
                        break
                lines = lines[:ending_line+1]
                voice_contents[0] = "\n".join(lines) + "\n"
                num_measures = [count_measure(content) for content in voice_contents]
                ending = re.findall(rf"\|(.[^\|]*%\d+\n)", voice_contents[0])[-1]
            else:
                # Multiple voice, truncate or fill the last voice to match the first voice
                num_measures = [count_measure(content) for content in voice_contents[:-1]]
                cur_gen_measures = count_measure(voice_contents[-1])
                ending = re.findall(rf"\|(.[^\|]*%\d+\n)", voice_contents[0])[-1]
                if cur_gen_measures > num_measures[0]:
                    lines = voice_contents[-1].split("\n")
                    new_voice_content = "\n".join(lines[:(num_measures[0]-1)//5])
                    new_voice_content += '\n'
                    r_measure = num_measures[0] % 5
                    if r_measure == 0:
                        r_measure = 5
                    cnt = 0
                    for c in lines[(num_measures[0]-1)//5]:
                        if c == '|':
                            cnt += 1
                        new_voice_content += c
                        if cnt == r_measure:
                            break
                    new_voice_content += ending
                    voice_contents[-1] = new_voice_content
                else:
                    rest = f"x{num_8th}"
                    while cur_gen_measures + 5 <= num_measures[0]:
                        voice_contents[-1] += f" {rest} | {rest} | {rest} | {rest} | {rest} | %{cur_gen_measures + 5}\n"
                        cur_gen_measures += 5
                    while cur_gen_measures < num_measures[0]:
                        voice_contents[-1] += f" {rest} |"
                        cur_gen_measures += 1
                    voice_contents[-1] += ending

            while len(voice_contents) < total_voices:
                voice_contents.append(generate_dummy_voice(num_measures[0], num_8th, ending))
            abc = header 
            for i in range(1, total_voices+1):
                abc += f"V:{i}\n{voice_contents[i-1]}"
    
    return abc

def convert_abc_to_xml(preds, tmp_dir):
    xmls = []
    tmp_musicxml_file = os.path.join(tmp_dir, "example.xml")
    for abc in tqdm(preds['abc_transcription'], desc="Converting ABC to XML..."):

        clean_abc = cleanup_abc(abc)
            
        result = subprocess.run(
            ABC2XML_CMD + ['-'], 
            input=clean_abc.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        if result.returncode != 0:
            print("ABC to XML conversion failed. Use empty file instead.")

        with open(tmp_musicxml_file, "w") as f:
            f.write(result.stdout.decode('utf-8'))

        result = subprocess.run(
            XMLFIX_CMD + [tmp_musicxml_file, '-o', tmp_musicxml_file],
            stderr=subprocess.PIPE,
            check=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"MuseScore XML formatting failed! Error: {result.stderr.decode('utf-8')}")

        with open(tmp_musicxml_file, "r") as f:
            xmls.append(f.read())

    return xmls

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file with transcriptions")
    parser.add_argument("--input_format", type=str, default="auto", help="Input format (default to auto-detect from input file)")
    parser.add_argument("--tmp_dir", type=str, default="/dev/shm/tmp", help="Temporary directory for intermediate files")

    args = parser.parse_args()

    if args.input_format == "auto":
        args.input_format = args.input_file.split('.')[0].split('_')[-1]

    os.makedirs(args.tmp_dir, exist_ok=True)

    with open(args.input_file, 'r') as f:
        preds = json.load(f)

    if args.input_format == "abc":
        outputs = convert_abc_to_xml(preds, args.tmp_dir)
        output_file = args.input_file.replace('_abc.json', '_xml.json')
    else:
        raise NotImplementedError(f"Input format {args.input_format} not supported")
        
    with open(output_file, "w") as f:
        json.dump(outputs, f)

    print(f"Saved XML outputs to {output_file}")
