#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Check sign initialization and estimate first-block attention in f64."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=root / 'target/grok-div97-search/qk-sign-initialization')
parser.add_argument('--binary', type=Path, default=root / 'target/release/mesh-gpt')
arguments = parser.parse_args()
study = json.loads((root / 'experiments/div97-mesh-calibrated-search.json').read_text())
binary = arguments.binary.resolve()
output = arguments.output.resolve()
output.mkdir(parents=True, exist_ok=True)
runs = []
parameters = []
for name, signs in [('unsigned', 0), ('signed', 1)]:
    settings = {**study['common'], **{k:v for k,v in study['candidates'][0].items() if k != 'label'},
                'batch':8, 'report-every':1, 'learning-rate':1e-30, 'qk-signs':signs}
    directory = output / name
    assert not directory.exists(), directory
    command = [str(binary), 'grok', '--steps', '1']
    for key, value in settings.items():
        command += ['--' + key, str(value)]
    command += ['--output', str(directory)]
    print('Capturing ' + name + ' initialization', flush=True)
    with (output / (name + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = (directory / 'model.bin').read_bytes()
    assert data[:8] == b'MSHGPT07'
    config = struct.unpack_from('<7I', data, 8)
    assert config == (99, 4, 128, 4, 2, 8, 16), config
    count = 26624
    offset = 36 + 4 * config[0]
    assert len(data) == offset + 4 * count
    values = struct.unpack_from(f'<{count}f', data, offset)
    parameters.append(values)
    with (directory / 'loss.csv').open() as source:
        history = list(csv.DictReader(source))
    assert len(history) == 2
    assert history[0]['train_cross_entropy'] == history[1]['train_cross_entropy']
    assert history[0]['validation_cross_entropy'] == history[1]['validation_cross_entropy']
    runs.append({'name':name, 'command':command, 'settings':settings,
                 'checkpoint_sha256':hashlib.sha256(data).hexdigest(),
                 'initial_training_cross_entropy':float(history[0]['train_cross_entropy']),
                 'initial_heldout_cross_entropy':float(history[0]['validation_cross_entropy'])})

assert parameters[0] != parameters[1]
assert all(abs(a) == abs(b) for a,b in zip(*parameters)), 'Magnitude changed'
width, heads, depth = 128, 4, 8
mesh_size, block_size = 2048, 12288
allowed = set()
for block in range(2):
    for projection in range(2):
        end = mesh_size + block * block_size + (projection + 1) * mesh_size
        allowed.update(range(end - 2 * width, end))
changed = {i for i,(a,b) in enumerate(zip(*parameters)) if a != b}
assert changed <= allowed and changed

bits = 7
masks = [1 << bit for bit in range(bits)] + [m for m in range(1,1 << bits) if m.bit_count() != 1]
def code(token):
    return [1.0 if (token & masks[c % len(masks)]).bit_count() % 2 else -1.0 for c in range(width)]

def mesh(x, weights, offset):
    for layer in range(depth):
        base = offset + 2 * width * layer
        x = [weights[base+2*c] * x[c] + weights[base+2*c+1] * x[(c+1) % width]
             for c in range(width)]
        if layer + 1 != depth:
            x = [v if v >= 0 else math.expm1(v) for v in x]
    return x

def normalize(x, groups):
    size = width // groups
    result = []
    for start in range(0,width,size):
        row = x[start:start+size]
        rms = math.sqrt(sum(v*v for v in row) / size + 1e-5)
        result += [v/rms for v in row]
    return result

def rotate(x, position):
    result = list(x)
    size = width // heads
    for start in range(0,width,size):
        for pair in range(size // 2):
            angle = position * 10000 ** (-2 * pair / size)
            cosine, sine = math.cos(angle), math.sin(angle)
            a, b = x[start+pair], x[start+pair+size//2]
            result[start+pair] = a*cosine - b*sine
            result[start+pair+size//2] = a*sine + b*cosine
    return result

with (output / 'unsigned/equations.csv').open() as source:
    equations = [[int(row[key]) for key in ('input_a_id','operator_id','input_b_id','equals_id')]
                 for row in csv.DictReader(source)]
assert len(equations) == 9312
for row, weights in zip(runs, parameters):
    inputs = [normalize(mesh(code(token),weights,0),1) for token in range(99)]
    query = normalize(mesh(inputs[98],weights,mesh_size),heads)
    keys = [normalize(mesh(x,weights,2*mesh_size),heads) for x in inputs]
    size = width // heads
    row['same_position_query_key_cosines'] = [
        sum(query[c] * keys[98][c] for c in range(start,start+size)) /
        math.sqrt(sum(query[c]**2 for c in range(start,start+size)) *
                  sum(keys[98][c]**2 for c in range(start,start+size)))
        for start in range(0,width,size)]
    query = rotate(query,3)
    scores = {}
    for token in range(99):
        for position in range(4):
            key = rotate(keys[token],position)
            scores[token,position] = [sum(query[c] * key[c] for c in range(start,start+size)) / math.sqrt(size)
                                      for start in range(0,width,size)]
    probabilities = [0.0] * heads
    for tokens in equations:
        for head in range(heads):
            logits = [scores[token,position][head] for position,token in enumerate(tokens)]
            maximum = max(logits)
            exp = [math.exp(value - maximum) for value in logits]
            probabilities[head] += exp[-1] / sum(exp)
    row['mean_first_block_self_attention_per_head'] = [p/len(equations) for p in probabilities]
    row['mean_first_block_self_attention'] = sum(probabilities) / (heads * len(equations))
    print(json.dumps({k:v for k,v in row.items() if k not in ('command','settings')}), flush=True)

invalid = subprocess.run([str(binary),'grok','--qk-signs','2'],capture_output=True,text=True)
assert invalid.returncode and 'Q/K sign initialization must be 0 or 1' in invalid.stderr
report = {'purpose':'first-block initialization analysis only; not grokking',
          'arithmetic':'independent Python f64 reconstruction of embeddings, RMSNorm, Q/K, RoPE and attention',
          'capture':'one FP32 update at learning rate 1e-30; observed losses unchanged',
          'equations':len(equations), 'parameters':26624,
          'magnitudes_identical':True, 'only_final_qk_rows_changed':True,
          'changed_parameters':len(changed),
          'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(), 'runs':runs}
(output / 'summary.json').write_text(json.dumps(report,indent=2) + '\n')
print('Initialization scope, magnitude and invalid-sign checks passed',flush=True)
