#!/usr/bin/env python3
"""
scripts/evaluate_quantitative.py
Performs complete target-anchored quantitative evaluation and dual-stream ablation
(Pure X-CLIP vs. Full Pipeline [+VLM]) from exported results in results/.

Strictly uses taxonomy, normalization, and path mappings from config.py.
"""

import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ALL_CANONICAL_CLASSES,
    EXCLUDED_BENCHMARK_CLASSES,
    KNOWN_CAUSAL_CO_OCCURRENCES,
    CLASS_NORMALIZATION_MAP,
    normalize_class_name,
    parse_timestamp,
    LABELS_DIR,
    TIMELINES_DIR,
    PREDICTIONS_JSON,
    OUTPUT_METRICS_JSON,
    OUTPUT_CONFUSION_CSV
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("QuantitativeEvaluator")


@dataclass
class GroundTruthEvent:
    video_id: str
    start_time: float
    end_time: float
    activity_class: str
    duration_sec: float = 0.0

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def center_time(self) -> float:
        return (self.start_time + self.end_time) / 2.0


@dataclass
class PredictedEvent:
    video_id: str
    start_time: float
    end_time: float
    activity_class: str
    confidence: float
    ranking_score: float = 0.0
    vlm_decision: str = "PENDING"
    vlm_evidence: Optional[str] = None

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def center_time(self) -> float:
        return (self.start_time + self.end_time) / 2.0


def compute_temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(1e-6, max(e1, e2) - min(s1, s2))
    return float(inter / union)


@dataclass
class MatchPair:
    pred: PredictedEvent
    gt: GroundTruthEvent
    tiou: float
    center_dist: float
    is_same_class: bool


class AdaptiveEventMatcher:
    """Matches predictions to ground truth events with optional vicinity relaxation."""
    @staticmethod
    def match(
        predictions: List[PredictedEvent],
        ground_truths: List[GroundTruthEvent],
        tiou_threshold: float = 0.5,
        max_center_distance: Optional[float] = None,
        require_same_class: bool = True
    ) -> Tuple[List[MatchPair], List[PredictedEvent], List[GroundTruthEvent]]:
        if not predictions:
            return [], [], list(ground_truths)
        if not ground_truths:
            return [], list(predictions), []

        sorted_preds = sorted(predictions, key=lambda p: (p.confidence, p.ranking_score), reverse=True)
        matched_pairs: List[MatchPair] = []
        matched_gts: Set[int] = set()
        matched_preds: Set[int] = set()

        for p_idx, p in enumerate(sorted_preds):
            best_gt_idx, best_score, best_tiou, best_cdist = None, -1.0, 0.0, 9999.0
            for g_idx, g in enumerate(ground_truths):
                if g_idx in matched_gts or (require_same_class and p.activity_class != g.activity_class):
                    continue

                tiou = compute_temporal_iou(p.start_time, p.end_time, g.start_time, g.end_time)
                cdist = abs(p.center_time() - g.center_time())

                qualifies = (tiou >= tiou_threshold) if max_center_distance is None else (tiou >= tiou_threshold or cdist <= max_center_distance)
                if qualifies:
                    score = tiou + (1.0 / (1.0 + cdist))
                    if score > best_score:
                        best_score, best_gt_idx, best_tiou, best_cdist = score, g_idx, tiou, cdist

            if best_gt_idx is not None:
                matched_gts.add(best_gt_idx)
                matched_preds.add(p_idx)
                matched_pairs.append(MatchPair(
                    p, ground_truths[best_gt_idx], best_tiou, best_cdist,
                    p.activity_class == ground_truths[best_gt_idx].activity_class
                ))

        unmatched_preds = [p for i, p in enumerate(sorted_preds) if i not in matched_preds]
        unmatched_gts = [g for i, g in enumerate(ground_truths) if i not in matched_gts]
        return matched_pairs, unmatched_preds, unmatched_gts


def load_ground_truth(labels_dir: Path) -> Dict[str, List[GroundTruthEvent]]:
    gt_by_video: Dict[str, List[GroundTruthEvent]] = {}
    if not labels_dir.is_dir():
        logger.warning(f"Labels directory '{labels_dir}' does not exist.")
        return gt_by_video

    for jf in labels_dir.glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        vid_name = data.get("video_filename", jf.stem + ".mp4")
        dur = parse_timestamp(data.get("duration_sec"), 100.0)
        events = []
        for ev in data.get("ground_truth", []):
            s = parse_timestamp(ev.get("start"), 0.0)
            e = parse_timestamp(ev.get("end"), dur)
            if ev.get("start") is None and ev.get("end") is None:
                s, e = 0.0, dur
            events.append(GroundTruthEvent(
                vid_name, min(s, e), max(s, e),
                normalize_class_name(ev.get("label", data.get("category", "Unknown"))),
                dur
            ))
        gt_by_video[vid_name] = events
        gt_by_video[Path(vid_name).stem] = events
    return gt_by_video


def load_dual_predictions(
    benchmark_json: Path,
    timelines_dir: Path
) -> Tuple[Dict[str, List[PredictedEvent]], Dict[str, List[PredictedEvent]], Dict[str, int]]:
    """Loads Full Pipeline predictions from JSON and Pure X-CLIP predictions from proposal diagnostics."""
    full_preds: Dict[str, List[PredictedEvent]] = {}
    xclip_preds: Dict[str, List[PredictedEvent]] = {}
    vlm_audit = {"SUPPORTED": 0, "UNCERTAIN": 0, "REJECTED": 0, "TOTAL": 0}

    # Full pipeline
    if benchmark_json.is_file():
        with open(benchmark_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for vid_name, acts in data.items():
            evs = [
                PredictedEvent(
                    vid_name,
                    parse_timestamp(a.get("start_time")),
                    parse_timestamp(a.get("end_time")),
                    normalize_class_name(a.get("class")),
                    float(a.get("confidence", 1.0)),
                    float(a.get("ranking_score", 0.0)),
                    a.get("vlm_decision", "SUPPORTED"),
                    a.get("vlm_evidence")
                )
                for a in acts
            ]
            full_preds[vid_name] = evs
            full_preds[Path(vid_name).stem] = evs

    # Pure X-CLIP proposals
    if timelines_dir.is_dir():
        for pf in timelines_dir.glob("*_proposals.json"):
            vid_stem = pf.name.replace("_proposals.json", "")
            vid_name = f"{vid_stem}.mp4"
            with open(pf, "r", encoding="utf-8") as f:
                pdata = json.load(f)

            evs_xclip = []
            for p in pdata.get("proposals", []):
                for hyp in p.get("hypotheses", []):
                    dec = hyp.get("vlm_decision", "PENDING")
                    vlm_audit[dec] = vlm_audit.get(dec, 0) + 1
                    vlm_audit["TOTAL"] += 1
                    evs_xclip.append(PredictedEvent(
                        vid_name,
                        parse_timestamp(p.get("start_time")),
                        parse_timestamp(p.get("end_time")),
                        normalize_class_name(hyp.get("class")),
                        float(hyp.get("confidence", 1.0)),
                        float(hyp.get("ranking_score", 0.0)),
                        dec,
                        hyp.get("vlm_evidence")
                    ))
            xclip_preds[vid_name] = evs_xclip
            xclip_preds[vid_stem] = evs_xclip

    return full_preds, xclip_preds, vlm_audit


class TargetAnchoredEvaluator:
    def __init__(
        self,
        gt_data: Dict[str, List[GroundTruthEvent]],
        pred_data: Dict[str, List[PredictedEvent]],
        strict_tiou: float = 0.5,
        lenient_tiou: float = 0.1,
        vicinity_dist: float = 5.0
    ):
        self.gt_data = gt_data
        self.pred_data = pred_data
        self.strict_tiou = strict_tiou
        self.lenient_tiou = lenient_tiou
        self.vicinity_dist = vicinity_dist

        # Canonical active benchmark classes
        present_gt = {ev.activity_class for evs in gt_data.values() for ev in evs if ev.activity_class in ALL_CANONICAL_CLASSES}
        self.active_classes = sorted(list(present_gt - EXCLUDED_BENCHMARK_CLASSES))
        self.video_names = sorted(list(set(k for k in gt_data.keys() if k.endswith(".mp4"))))

    def evaluate(self) -> Dict[str, Any]:
        strict_tp, strict_fp, strict_fn = [], [], []
        vicinity_tp, vicinity_fp, vicinity_fn = [], [], []
        all_gts, all_preds = [], []

        anchored_hits, total_gt_videos = 0, 0
        true_false_alarms, co_occurring_extras = [], []
        per_class_stats = {c: {"gt_vids": 0, "pred_vids": 0, "tp": 0, "fp": 0, "fn": 0} for c in ALL_CANONICAL_CLASSES}
        conf_matches, conf_fps, conf_fns = [], [], []

        for vid in self.video_names:
            v_gts = self.gt_data.get(vid, self.gt_data.get(Path(vid).stem, []))
            v_preds = self.pred_data.get(vid, self.pred_data.get(Path(vid).stem, []))
            all_gts.extend(v_gts)
            all_preds.extend(v_preds)

            gt_classes = set(g.activity_class for g in v_gts if g.activity_class in ALL_CANONICAL_CLASSES)
            pred_classes = set(p.activity_class for p in v_preds if p.activity_class in ALL_CANONICAL_CLASSES)

            # Strict and vicinity matches
            s_tp, s_fp, s_fn = AdaptiveEventMatcher.match(v_preds, v_gts, self.strict_tiou, require_same_class=True)
            v_tp, v_fp, v_fn = AdaptiveEventMatcher.match(v_preds, v_gts, self.lenient_tiou, self.vicinity_dist, require_same_class=True)
            strict_tp.extend(s_tp)
            strict_fp.extend(s_fp)
            strict_fn.extend(s_fn)
            vicinity_tp.extend(v_tp)
            vicinity_fp.extend(v_fp)
            vicinity_fn.extend(v_fn)

            # Class-agnostic confusion matching on remaining intervals
            matched_pred_ids = set(id(m.pred) for m in v_tp)
            matched_gt_ids = set(id(m.gt) for m in v_tp)
            c_matches, c_fps, c_fns = AdaptiveEventMatcher.match(
                [p for p in v_preds if id(p) not in matched_pred_ids],
                [g for g in v_gts if id(g) not in matched_gt_ids],
                0.1,
                self.vicinity_dist,
                require_same_class=False
            )
            conf_matches.extend(v_tp + c_matches)
            conf_fps.extend(c_fps)
            conf_fns.extend(c_fns)

            # Target-anchored evaluation
            if gt_classes:
                total_gt_videos += 1
                target_found = len(gt_classes.intersection(pred_classes)) > 0
                if target_found:
                    anchored_hits += 1
                    for p in v_preds:
                        if p.activity_class not in gt_classes:
                            co_occurring_extras.append({
                                "video_id": vid,
                                "primary_gt": list(gt_classes),
                                "extra_activity": p.activity_class,
                                "interval": [p.start_time, p.end_time],
                                "confidence": round(p.confidence, 4)
                            })
                else:
                    for p in v_preds:
                        true_false_alarms.append({
                            "video_id": vid,
                            "expected_gt": list(gt_classes),
                            "hallucinated_class": p.activity_class,
                            "interval": [p.start_time, p.end_time],
                            "confidence": round(p.confidence, 4)
                        })

                for c in ALL_CANONICAL_CLASSES:
                    in_g, in_p = c in gt_classes, c in pred_classes
                    if in_g:
                        per_class_stats[c]["gt_vids"] += 1
                    if in_p:
                        per_class_stats[c]["pred_vids"] += 1
                    if in_g and in_p:
                        per_class_stats[c]["tp"] += 1
                    elif in_g and not in_p:
                        per_class_stats[c]["fn"] += 1
                    elif not in_g and in_p and not target_found:
                        per_class_stats[c]["fp"] += 1

        final_class_metrics = {}
        for c, st in per_class_stats.items():
            p = st["tp"] / max(1, st["tp"] + st["fp"])
            r = st["tp"] / max(1, st["tp"] + st["fn"])
            f1 = (2 * p * r) / max(1e-6, p + r)
            final_class_metrics[c] = {
                "gt_videos": st["gt_vids"],
                "pred_videos": st["pred_vids"],
                "video_precision": round(p, 4),
                "video_recall": round(r, 4),
                "video_f1": round(f1, 4)
            }

        active_f1s = [final_class_metrics[c]["video_f1"] for c in self.active_classes if final_class_metrics[c]["gt_videos"] > 0]
        macro_f1 = float(sum(active_f1s) / len(active_f1s)) if active_f1s else 0.0

        # Boundary MAE
        s_errs = [abs(m.pred.start_time - m.gt.start_time) for m in vicinity_tp]
        e_errs = [abs(m.pred.end_time - m.gt.end_time) for m in vicinity_tp]
        b_mae = float(sum(s_errs + e_errs) / max(1, len(s_errs) * 2)) if s_errs else 0.0

        # Confusion Matrix
        conf_matrix, t_conf, t_causal = self._build_confusion_matrix(conf_matches, conf_fps, conf_fns)

        total_vids = max(1, total_gt_videos)
        hit_acc = anchored_hits / total_vids
        a_prec = anchored_hits / max(1, anchored_hits + len(true_false_alarms))
        a_f1 = (2 * a_prec * hit_acc) / max(1e-6, a_prec + hit_acc)

        return {
            "summary": {
                "evaluated_videos": len(self.video_names),
                "gt_videos": total_gt_videos,
                "total_ground_truths": len(all_gts),
                "total_predictions": len(all_preds),
                "target_anchored_hits": anchored_hits,
                "true_false_alarms_count": len(true_false_alarms),
                "co_occurring_extras_count": len(co_occurring_extras)
            },
            "target_anchored_evaluation": {
                "hit_accuracy": round(hit_acc, 4),
                "anchored_precision": round(a_prec, 4),
                "anchored_f1_score": round(a_f1, 4),
                "macro_f1_score": round(macro_f1, 4),
                "per_class_metrics": final_class_metrics
            },
            "temporal_localization": {
                "strict_tp": len(strict_tp),
                "strict_fp": len(strict_fp),
                "strict_fn": len(strict_fn),
                "strict_precision_05": round(len(strict_tp) / max(1, len(strict_tp) + len(strict_fp)), 4),
                "strict_recall_05": round(len(strict_tp) / max(1, len(strict_tp) + len(strict_fn)), 4),
                "strict_f1_05": round(2 * len(strict_tp) / max(1, 2 * len(strict_tp) + len(strict_fp) + len(strict_fn)), 4),
                "vicinity_tp": len(vicinity_tp),
                "vicinity_precision": round(len(vicinity_tp) / max(1, len(vicinity_tp) + len(vicinity_fp)), 4),
                "vicinity_recall": round(len(vicinity_tp) / max(1, len(vicinity_tp) + len(vicinity_fn)), 4),
                "vicinity_f1": round(2 * len(vicinity_tp) / max(1, 2 * len(vicinity_tp) + len(vicinity_fp) + len(vicinity_fn)), 4),
                "mean_boundary_mae_sec": round(b_mae, 2)
            },
            "confusion_and_overlap": {
                "top_confused_pairs": t_conf,
                "co_occurring_causal_pairs": t_causal,
                "interval_confusion_matrix": conf_matrix
            }
        }

    def _build_confusion_matrix(self, matches: List[MatchPair], fps: List[PredictedEvent], fns: List[GroundTruthEvent]):
        classes = list(ALL_CANONICAL_CLASSES)
        matrix = {r: {c: 0 for c in classes + ["Background (Missed)"]} for r in classes + ["Background (Spurious)"]}
        true_conf_counts, causal_counts = {}, {}

        for m in matches:
            gt_c = m.gt.activity_class if m.gt.activity_class in matrix else "Unknown"
            pred_c = m.pred.activity_class if m.pred.activity_class in classes else "Unknown"
            if gt_c in matrix and pred_c in matrix[gt_c]:
                matrix[gt_c][pred_c] += 1
            if gt_c != pred_c and gt_c != "Unknown" and pred_c != "Unknown":
                pair = (gt_c, pred_c)
                if pair in KNOWN_CAUSAL_CO_OCCURRENCES:
                    causal_counts[pair] = causal_counts.get(pair, 0) + 1
                else:
                    true_conf_counts[pair] = true_conf_counts.get(pair, 0) + 1

        for g in fns:
            if g.activity_class in matrix:
                matrix[g.activity_class]["Background (Missed)"] += 1
        for p in fps:
            if p.activity_class in matrix["Background (Spurious)"]:
                matrix["Background (Spurious)"][p.activity_class] += 1

        t_conf = [{"actual_class": k[0], "predicted_class": k[1], "count": v} for k, v in sorted(true_conf_counts.items(), key=lambda x: x[1], reverse=True)]
        t_causal = [{"actual_class": k[0], "predicted_class": k[1], "count": v} for k, v in sorted(causal_counts.items(), key=lambda x: x[1], reverse=True)]
        return matrix, t_conf, t_causal


def main():
    logger.info("Loading ground truth labels...")
    gt_data = load_ground_truth(LABELS_DIR)
    logger.info(f"Loaded ground truth for {len(gt_data) // 2} video(s) from '{LABELS_DIR}'.")

    logger.info("Loading prediction streams (Full Pipeline & Pure X-CLIP)...")
    full_preds, xclip_preds, vlm_audit = load_dual_predictions(PREDICTIONS_JSON, TIMELINES_DIR)
    logger.info(f"Loaded predictions: Full Pipeline={len(full_preds) // 2 if full_preds else 0} vids, Pure X-CLIP={len(xclip_preds) // 2 if xclip_preds else 0} vids.")

    res_xclip = TargetAnchoredEvaluator(gt_data, xclip_preds).evaluate()
    res_full = TargetAnchoredEvaluator(gt_data, full_preds).evaluate()

    # Terminal Reporting
    print("\n" + "=" * 80)
    print("      TARGET-ANCHORED EVALUATION & ABLATION (PURE X-CLIP vs. FULL PIPELINE)")
    print("=" * 80)
    print(f"{'Performance Metric':<38} | {'Pure X-CLIP':<14} | {'Full (+VLM)':<14} | {'Delta':<6}")
    print("-" * 80)

    comparisons = [
        ("Target-Anchored Hit Accuracy", res_xclip["target_anchored_evaluation"]["hit_accuracy"], res_full["target_anchored_evaluation"]["hit_accuracy"], True),
        ("Target-Anchored Precision", res_xclip["target_anchored_evaluation"]["anchored_precision"], res_full["target_anchored_evaluation"]["anchored_precision"], True),
        ("Target-Anchored F1-Score", res_xclip["target_anchored_evaluation"]["anchored_f1_score"], res_full["target_anchored_evaluation"]["anchored_f1_score"], True),
        ("Macro F1-Score (Active Classes)", res_xclip["target_anchored_evaluation"]["macro_f1_score"], res_full["target_anchored_evaluation"]["macro_f1_score"], True),
        ("Genuine False Alarms (Target Missing)", res_xclip["summary"]["true_false_alarms_count"], res_full["summary"]["true_false_alarms_count"], False),
        ("Precursor/Co-Occurring Actions", res_xclip["summary"]["co_occurring_extras_count"], res_full["summary"]["co_occurring_extras_count"], False),
        ("Strict Precision (tIoU >= 0.5)", res_xclip["temporal_localization"]["strict_precision_05"], res_full["temporal_localization"]["strict_precision_05"], True),
        ("Strict Recall (tIoU >= 0.5)", res_xclip["temporal_localization"]["strict_recall_05"], res_full["temporal_localization"]["strict_recall_05"], True),
        ("Strict F1-Score (tIoU >= 0.5)", res_xclip["temporal_localization"]["strict_f1_05"], res_full["temporal_localization"]["strict_f1_05"], True),
        ("Vicinity Precision (tIoU>=0.1 / Center<=5s)", res_xclip["temporal_localization"]["vicinity_precision"], res_full["temporal_localization"]["vicinity_precision"], True),
        ("Vicinity Recall (Semantic Hit)", res_xclip["temporal_localization"]["vicinity_recall"], res_full["temporal_localization"]["vicinity_recall"], True),
        ("Vicinity F1-Score", res_xclip["temporal_localization"]["vicinity_f1"], res_full["temporal_localization"]["vicinity_f1"], True),
        ("Mean Boundary MAE (seconds)", res_xclip["temporal_localization"]["mean_boundary_mae_sec"], res_full["temporal_localization"]["mean_boundary_mae_sec"], True),
        ("Total Predictions Generated", res_xclip["summary"]["total_predictions"], res_full["summary"]["total_predictions"], False)
    ]

    for label, vx, vf, is_float in comparisons:
        diff = vf - vx
        sign = "+" if diff > 0 else ""
        if is_float:
            print(f"{label:<38} | {vx:<14.4f} | {vf:<14.4f} | {sign}{diff:<6.4f}")
        else:
            print(f"{label:<38} | {vx:<14} | {vf:<14} | {sign}{diff:<6}")

    print("=" * 80)
    print(f"VLM Decision Audit: Total Screened: {vlm_audit['TOTAL']} | Supported: {vlm_audit.get('SUPPORTED', 0)} | Uncertain: {vlm_audit.get('UNCERTAIN', 0)} | Rejected: {vlm_audit.get('REJECTED', 0)}")
    print("=" * 80)

    # Class Breakdown
    print("\n" + "=" * 80)
    print("         PER-CLASS VIDEO ACCURACY (ACTIVE BENCHMARK CLASSES ONLY)")
    print("=" * 80)
    print(f"{'Activity Class':<35} | {'GT Vids':<8} | {'Pred Vids':<10} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
    print("-" * 80)
    per_class = res_full["target_anchored_evaluation"]["per_class_metrics"]
    for c in ALL_CANONICAL_CLASSES:
        if c in EXCLUDED_BENCHMARK_CLASSES:
            continue
        st = per_class.get(c, {})
        print(f"{c:<35} | {st.get('gt_videos', 0):<8} | {st.get('pred_videos', 0):<10} | {st.get('video_precision', 0.0):<7.4f} | {st.get('video_recall', 0.0):<7.4f} | {st.get('video_f1', 0.0):<7.4f}")
    print("=" * 80)

    # Export metrics JSON & confusion CSV into results/
    OUTPUT_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({"vlm_decision_audit": vlm_audit, "pure_xclip": res_xclip, "full_pipeline": res_full}, f, indent=2)
    logger.info(f"Saved quantitative metrics payload to '{OUTPUT_METRICS_JSON}'.")

    matrix = res_full["confusion_and_overlap"]["interval_confusion_matrix"]
    cols = list(matrix[list(matrix.keys())[0]].keys())
    csv_lines = ["," + ",".join(f'"{c}"' for c in cols)]
    for r in matrix:
        csv_lines.append(f'"{r}",' + ",".join(str(matrix[r][c]) for c in cols))
    OUTPUT_CONFUSION_CSV.write_text("\n".join(csv_lines), encoding="utf-8")
    logger.info(f"Saved confusion matrix CSV to '{OUTPUT_CONFUSION_CSV}'.")


if __name__ == "__main__":
    main()
