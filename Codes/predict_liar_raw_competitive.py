# encoding: utf-8
import argparse
import json
import os
import sys
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(pjoin(os.path.dirname(os.path.abspath(__file__)), "helpers"))
sys.path.append(pjoin(os.path.dirname(os.path.abspath(__file__)), "model"))

from helpers.reader5 import COMPETITIVE_FEATURE_NAMES, myDataset

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_IDS = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}
ID_TO_LABEL = {value: key for key, value in LABEL_IDS.items()}


def load_model(model_path, device):
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)
    if not hasattr(model, "use_competitive_features"):
        model.use_competitive_features = False
    model = model.to(device)
    model.eval()
    return model


def label_prob_dict(probs):
    return {ID_TO_LABEL[index]: float(probs[index]) for index in range(len(ID_TO_LABEL))}


def prediction_explanation(predicted_label, feature_row):
    support_strength = float(feature_row.get("support_strength", 0.0) or 0.0)
    refute_strength = float(feature_row.get("refute_strength", 0.0) or 0.0)
    support_evidence = feature_row.get("support_evidence", []) or []
    refute_evidence = feature_row.get("refute_evidence", []) or []
    neutral_evidence = feature_row.get("neutral_evidence", []) or []

    if refute_strength > support_strength:
        direction = "refuting"
        candidates = refute_evidence
    elif support_strength > refute_strength:
        direction = "supporting"
        candidates = support_evidence
    else:
        direction = "neutral or balanced"
        candidates = neutral_evidence

    domains = []
    for candidate in candidates:
        domain = candidate.get("source_domain")
        if domain and domain not in domains:
            domains.append(domain)
    source_text = ", ".join(domains[:3]) if domains else "no selected sources"
    strongest = max(candidates, key=lambda item: float(item.get("final_weight", 0.0) or 0.0), default={})
    sentence = strongest.get("sentence", "No decisive evidence was selected.")
    return (
        f"Prediction: {predicted_label}. The selected evidence is {direction}: "
        f"support strength={support_strength:.3f}, refute strength={refute_strength:.3f}. "
        f"Main selected sources: {source_text}. Strongest selected sentence: {sentence}"
    )


def build_prediction_row(raw_text_dict, lm_ids_dict, probs, pred_id, row_index):
    feature_row = raw_text_dict.get("competitive_feature_rows", [{}])[row_index] or {}
    predicted_label = ID_TO_LABEL[int(pred_id)]
    output = {
        "event_id": raw_text_dict["event_id"][row_index],
        "claim": raw_text_dict["_CLAIM_TOK"][row_index],
        "predicted_label": predicted_label,
        "confidence": float(torch.max(probs).item()),
        "label_probs": label_prob_dict(probs.tolist()),
    }

    for name in COMPETITIVE_FEATURE_NAMES:
        output[name] = float(feature_row.get(name, 0.0) or 0.0)

    output["support_evidence"] = feature_row.get("support_evidence", [])
    output["refute_evidence"] = feature_row.get("refute_evidence", [])
    output["explanation"] = prediction_explanation(predicted_label, feature_row)
    return output


def predict(model, data_path, output_path, competitive_features_path=None, report_each_claim=30, limit=None):
    device = next(model.parameters()).device
    dataset = myDataset(
        data_path,
        report_each_claim=report_each_claim,
        competitive_features_path=competitive_features_path,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.my_collate)

    rows_written = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        with torch.no_grad():
            for oracle_ids, label_ids, raw_text_dict, lm_ids_dict in tqdm(loader):
                logits, _, _ = model.forward(oracle_ids, label_ids, lm_ids_dict)
                probs_batch = torch.softmax(logits, dim=-1).cpu()
                pred_ids = torch.argmax(probs_batch, dim=1).tolist()

                for row_index, pred_id in enumerate(pred_ids):
                    row = build_prediction_row(raw_text_dict, lm_ids_dict, probs_batch[row_index], pred_id, row_index)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                    if limit is not None and rows_written >= limit:
                        return rows_written
    return rows_written


def parse_args():
    parser = argparse.ArgumentParser(description="Export LIAR-RAW six-class fake-news predictions as JSONL.")
    parser.add_argument("--model", required=True, help="Path to a trained ExplainFC .pt model.")
    parser.add_argument("--data", default=pjoin(CODE_DIR, "dataset", "LIAR-RAW", "test.json"))
    parser.add_argument("--features", default=pjoin(CODE_DIR, "dataset", "features", "liar_raw_test_competitive.json"))
    parser.add_argument("--output", default=pjoin(CODE_DIR, "dataset", "features", "predictions_liar_raw_competitive.jsonl"))
    parser.add_argument("--report-each-claim", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    rows = predict(
        model,
        args.data,
        args.output,
        competitive_features_path=args.features,
        report_each_claim=args.report_each_claim,
        limit=args.limit,
    )
    print(f"Wrote {rows} prediction rows to {args.output}")


if __name__ == "__main__":
    main()
