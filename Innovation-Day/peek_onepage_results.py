import os
import re
import json
from collections import defaultdict

base = 'logs'

rows = []

for root, dirs, files in os.walk(base):
    for fname in files:
        if fname != 'result.json':
            continue
        path = os.path.join(root, fname)
        rel = os.path.relpath(path, base)

        m = re.search(r'CW_tam_p(\d+)_page(\d+)', rel)
        if not m:
            continue
        padl = int(m.group(1))
        page = int(m.group(2))

        parts = rel.split(os.sep)
        model = parts[1] if len(parts) > 1 else 'UNK'

        with open(path) as f:
            data = json.load(f)

        acc  = data.get('Accuracy')
        prec = data.get('Precision')
        rec  = data.get('Recall')
        f1   = data.get('F1-score')

        rows.append((padl, page, model, acc, prec, rec, f1))

# aggregate by (padl, model)
agg = defaultdict(lambda: {'n':0, 'acc':0.0, 'prec':0.0, 'rec':0.0, 'f1':0.0})

for padl, page, model, acc, prec, rec, f1 in rows:
    key = (padl, model)
    a = agg[key]
    a['n']   += 1
    a['acc'] += acc
    a['prec']+= prec
    a['rec'] += rec
    a['f1']  += f1

print("PadL,Model,Npages,Acc_mean,Prec_mean,Rec_mean,F1_mean")
for (padl, model) in sorted(agg.keys()):
    a = agg[(padl, model)]
    n = a['n']
    acc_mean  = a['acc']/n
    prec_mean = a['prec']/n
    rec_mean  = a['rec']/n
    f1_mean   = a['f1']/n
    print(f"{padl},{model},{n},{acc_mean:.6f},{prec_mean:.6f},{rec_mean:.6f},{f1_mean:.6f}")
