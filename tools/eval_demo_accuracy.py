"""Benchmark the current demo engine on annotated invoices from mcocr_train_df.csv."""
import csv
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from invoice_demo.engine import InvoiceEngine, LABELS, FIELD_NAMES

def norm(s):
    return " ".join(str(s).casefold().split())

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]

def evaluate_subset(num_samples=50, seed=42):
    csv_path = ROOT / "mcocr_train_df.csv"
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [r for r in reader if r.get("anno_texts") and r.get("anno_labels")]

    np.random.seed(seed)
    # Pick deterministic sample
    indices = np.random.RandomState(seed).permutation(len(all_rows))[:num_samples]
    sample_rows = [all_rows[i] for i in sorted(indices)]

    engine = InvoiceEngine()
    engine.load(lambda *args, **kwargs: None)

    results = []
    start_total = time.time()
    
    for idx, row in enumerate(sample_rows):
        img_id = row["img_id"]
        img_path = ROOT / "train_images/train_images" / img_id
        if not img_path.is_file():
            img_path = ROOT / "train_images" / img_id
        if not img_path.is_file():
            continue

        gt_texts = str(row["anno_texts"]).split("|||")
        gt_labels = str(row["anno_labels"]).split("|||")
        if len(gt_texts) != len(gt_labels):
            continue

        truth = {f: [] for f in LABELS}
        for t, l in zip(gt_texts, gt_labels):
            if l in truth:
                truth[l].append(t)

        try:
            pred = engine.process(img_path, lambda *args, **kwargs: None)
            pred_fields = {k: pred["fields"][k]["value"] for k in LABELS}
        except Exception as e:
            print(f"Error on {img_id}: {e}")
            continue

        for field in LABELS:
            ref = norm(" ".join(truth[field]))
            hyp = norm(pred_fields[field])
            ed = edit_distance(ref, hyp)
            max_len = max(len(ref), len(hyp))
            ncer = ed / max_len if max_len > 0 else 0.0
            cer = ed / len(ref) if len(ref) > 0 else (0.0 if len(hyp) == 0 else 1.0)
            exact_match = 1.0 if ref == hyp else 0.0
            
            # Fuzzy match (similarity ratio)
            from difflib import SequenceMatcher
            sim = SequenceMatcher(None, ref, hyp).ratio()

            # Confidence
            conf = pred["fields"][field].get("confidence", 0.0)

            results.append({
                "img_id": img_id,
                "field": field,
                "ref": ref,
                "hyp": hyp,
                "ref_len": len(ref),
                "hyp_len": len(hyp),
                "ed": ed,
                "cer": cer,
                "ncer": ncer,
                "exact_match": exact_match,
                "similarity": sim,
                "confidence": conf,
            })
        print(f"[{idx+1}/{len(sample_rows)}] Processed {img_id}")

    elapsed = time.time() - start_total
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS (Sample {num_samples} images, Elapsed: {elapsed:.1f}s)")
    print("="*70)
    
    summary = []
    for field in LABELS:
        f_df = df[df["field"] == field]
        ref_chars = f_df["ref_len"].sum()
        max_chars = f_df[["ref_len", "hyp_len"]].max(axis=1).sum()
        total_ed = f_df["ed"].sum()
        
        field_cer = total_ed / ref_chars if ref_chars > 0 else 0.0
        field_ncer = total_ed / max_chars if max_chars > 0 else 0.0
        field_acc = 1.0 - field_ncer
        field_em = f_df["exact_match"].mean() * 100
        field_sim = f_df["similarity"].mean() * 100
        field_conf = f_df["confidence"].mean()

        summary.append({
            "Field": field,
            "Images": len(f_df),
            "Exact Match (%)": round(field_em, 2),
            "Similarity (%)": round(field_sim, 2),
            "Char Acc (1-NCER) (%)": round(field_acc * 100, 2),
            "Mean Confidence (%)": round(field_conf, 1),
            "NCER": round(field_ncer, 4),
            "CER": round(field_cer, 4),
        })

    # MICRO
    ref_chars = df["ref_len"].sum()
    max_chars = df[["ref_len", "hyp_len"]].max(axis=1).sum()
    total_ed = df["ed"].sum()
    micro_cer = total_ed / ref_chars if ref_chars > 0 else 0.0
    micro_ncer = total_ed / max_chars if max_chars > 0 else 0.0
    micro_acc = 1.0 - micro_ncer
    micro_em = df["exact_match"].mean() * 100
    micro_sim = df["similarity"].mean() * 100
    micro_conf = df["confidence"].mean()

    summary.append({
        "Field": "OVERALL (MICRO)",
        "Images": len(df["img_id"].unique()),
        "Exact Match (%)": round(micro_em, 2),
        "Similarity (%)": round(micro_sim, 2),
        "Char Acc (1-NCER) (%)": round(micro_acc * 100, 2),
        "Mean Confidence (%)": round(micro_conf, 1),
        "NCER": round(micro_ncer, 4),
        "CER": round(micro_cer, 4),
    })

    sum_df = pd.DataFrame(summary)
    print(sum_df.to_string(index=False))
    
    # Save results
    out_dir = ROOT / "output/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "baseline_demo_details.csv", index=False, encoding="utf-8-sig")
    sum_df.to_csv(out_dir / "baseline_demo_summary.csv", index=False, encoding="utf-8-sig")
    return sum_df

if __name__ == "__main__":
    evaluate_subset(num_samples=30, seed=42)
