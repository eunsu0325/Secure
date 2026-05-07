"""Experiment 1 Baselines: MobileFaceNet/IR-ResNet50 + ArcFace on Tongji ROI.

This is an independent baseline track for protocol-analysis study (Protocol A/B/C).
See plans/you-are-helping-implement-witty-river.md for the full design.

Two hard rules:
  Rule 1: fixed tau NEVER uses external_test_ids or future_ids.
  Rule 2: known true accept is ALWAYS (top1_id == gt_id) AND (top1_score >= tau).
"""
