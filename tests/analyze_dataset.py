import csv
import glob
from collections import Counter
from pathlib import Path

csv_files = glob.glob("*.csv")
print("Found CSV files:", [Path(f).name for f in csv_files])

target_file = None
for f in csv_files:
    if "2000" in f or "3000" in f:
        target_file = f
        break

if not target_file and csv_files:
    target_file = csv_files[0]

if target_file:
    print(f"Reading: {target_file}")
    cats = Counter()
    total = 0
    samples = {}
    with open(target_file, "r", encoding="utf-8-sig", errors="replace") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            total += 1
            cat = row.get("التصنيف") or "غير مصنف"
            cats[cat] += 1
            if cat not in samples and (row.get("السؤال") or "").strip():
                samples[cat] = row.get("السؤال")

    print(f"Total rows: {total}")
    print("\nCategories breakdown:")
    for cat, count in cats.most_common():
        print(f"  - {cat}: {count} | Example: {samples.get(cat, '')}")
