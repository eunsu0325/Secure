"""Modular Protocol A/B/C evaluators with shared score matrix.

Architecture:
  1. score_matrix.py — build [N_query, N_proto_max] cosine matrix once
  2. metrics.py      — shared metric primitives (Wilson CI, TPIR, FPIR, calibration)
  3. protocol_a.py   — closed-set rank-1 (background)
  4. protocol_b.py   — static open-set TPIR@FPIR (single snapshot)
  5. protocol_c.py   — sequential open-set with C-fixed and C-recal variants
  6. orchestrator.py — run all protocols + order_seed loop, save results
"""
