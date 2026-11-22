"""Create blinded pairwise human-annotation sheets from revision results."""

import argparse
import csv
import itertools
import os
import random


def create_annotation_files(input_path, output_path, seed):
    with open(input_path, "r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No evaluation rows found in {input_path}")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)

    rng = random.Random(seed)
    annotations = []
    answer_key = []
    for query_id, candidates in grouped.items():
        if len(candidates) < 2:
            continue
        question = candidates[0]["question"]
        for pair_number, (first, second) in enumerate(
            itertools.combinations(candidates, 2),
            start=1,
        ):
            if rng.random() < 0.5:
                first, second = second, first
            pair_id = f"{query_id}-{pair_number}"
            annotations.append({
                "pair_id": pair_id,
                "question": question,
                "answer_a": first["answer"],
                "evidence_a": first["retrieved_context"],
                "answer_b": second["answer"],
                "evidence_b": second["retrieved_context"],
                "correctness_preference": "",
                "evidence_support_preference": "",
                "completeness_preference": "",
                "abstention_appropriate": "",
                "annotator_id": "",
                "notes": "",
            })
            answer_key.extend((
                {"pair_id": pair_id, "blind_label": "A", "system": first["system"]},
                {"pair_id": pair_id, "blind_label": "B", "system": second["system"]},
            ))

    if not annotations:
        raise ValueError("No query has at least two system outputs to compare.")

    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    key_path = os.path.join(output_dir, "human_eval_key.csv")
    for path, records in ((output_path, annotations), (key_path, answer_key)):
        with open(path, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    print(f"Wrote {len(annotations)} blinded comparisons to {output_path}")
    print(f"Keep the unblinding key separate: {key_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join("evaluation", "revision_results.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("evaluation", "human_eval.csv"),
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    create_annotation_files(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
