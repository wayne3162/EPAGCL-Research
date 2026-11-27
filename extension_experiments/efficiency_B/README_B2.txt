EPAGCL B2 grouped-matmul experiment package
===========================================

Use the existing working EPAGCL WSL / CUDA / uv Python environment.
No packages need to be installed.

WHAT B2 CHANGES
---------------
B2 extends the validated MyModelB1: full embeddings are normalized once.
Instead of performing 2 matrix multiplications per independent contrastive
batch, it stacks groups of 2, 4, or 8 original BxB similarities and computes
batched matrix multiplications (torch.bmm). It does NOT mix candidates across
original batches. Two directional calls still generate separate randperm(N).
The final incomplete batch (189 nodes for Coauthor-Phy, N=34493, B=256)
remains an exact 189x189 comparison. The original Model.py, B0/B1 code, VR
penalty, augmentation, optimizer, and evaluation data remain unchanged.

FILES
-----
Copy the following THREE .py files to:
  ~/projects/EPAGCL/extension_experiments/efficiency_B/

  model_b2.py          independent B2 model, depends on model_b1.py
  verify_b2.py         numerical and RNG equivalence with original B0
  benchmark_b012.py    B0/B1/B2x2/B2x4/B2x8 paired GPU step benchmark

Keep your original B1 files (model_b1.py, verify_b1.py,
benchmark_b01.py) and their already-passing equivalence.json.

STEP 1: B2 EQUIVALENCE (DO THIS FIRST)
-------------------------------------
cd ~/projects/EPAGCL
B1_VERIFY="extension_experiments/efficiency_B/outputs/20261120_172257_B1_verify_seed_20261123/equivalence.json"
python -u extension_experiments/efficiency_B/verify_b2.py \
    --b1-verification-json "$B1_VERIFY" \
    --device cuda:0 --seed 20261123 --batch-size 256

Expected output:
  extension_experiments/efficiency_B/outputs/<timestamp>_B2_verify_seed_20261123/equivalence_b2.json

Required before benchmarking: Overall: PASS, with group counts 2/4/8 all
passing every test. Input gradients: N=64,256,257,1025,34493.
Model parameter gradients with original VR: N=64,256,257,1025.
Unchanged two independent randperm draws and post-loss RNG states checked.
Strict tolerances (no relaxing): rtol=1e-4, atol=1e-5.

STEP 2: PAIRED TIMING (ONLY IF STEP 1 PASSES)
-------------------------------------------
cd ~/projects/EPAGCL
B1_VERIFY="extension_experiments/efficiency_B/outputs/20261120_172257_B1_verify_seed_20261123/equivalence.json"
B2_VERIFY=$(find extension_experiments/efficiency_B/outputs \
    -maxdepth 2 -name equivalence_b2.json -type f | sort | tail -n 1)
python -u extension_experiments/efficiency_B/benchmark_b012.py \
    --b1-verification-json "$B1_VERIFY" \
    --b2-verification-json "$B2_VERIFY" \
    --device cuda:0 --seed 20261123 --rounds 3 --warmup 5 --timed 20

Order by round:
  1: B0 / B1 / B2x2 / B2x4 / B2x8
  2: B2x8 / B2x4 / B2x2 / B1 / B0
  3: B0 / B1 / B2x2 / B2x4 / B2x8

Expected output:
  extension_experiments/efficiency_B/outputs/<timestamp>_B012_compare/comparison_b012.json
Includes individual step timings, mean/median/P90, peak CUDA allocated
memory, per-round B2 vs B0/B1 comparisons, and all source SHA256 hashes.

IMPORTANT RESEARCH CONDITIONS
-----------------------------
- Keep every trial and anomaly. Do not discard bad rounds silently.
- Timed steps exclude preprocessing, profiler and linear probe.
- B2 numerical / gradient equivalence does not prove matching 1000-epoch
  training trajectories or identical node classification accuracy.
- Do not run the paired benchmark when verification is FAIL.
- Do not alter original Model.py, data split files or formal_A results.
- You need actual CUDA GPU verification: author-side tests were CPU-only,
  using a FakeGCNConv for the small synthetic model-gradient test.
- If CUDA reports OOM or numerical mismatches, report JSON and terminal log.
- Previous B0 4.7-second anomaly did not reproduce at seed 20263123;
  preserve it in the record, and interpret paired runs separately.
