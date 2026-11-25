# Training-data distribution: findings (2026-11-23)

Goal: answer "what is a good dataset?" for the value net. Findings from comparing the
training data with the positions the search actually evaluates. Nothing has been changed
in training yet; this is the starting point for later value-net experiments.

## Files and tools

| What | Path |
|---|---|
| raw reservoir data | `/mnt/c/TrainingData/reservoir-test.samples` (357,537,180 positions) |
| deduplicated | `/mnt/c/TrainingData/reservoir-full.dedup.samples` (252,455,875) |
| old rescored (current value-net data) | `/mnt/c/TrainingData/reservoir-test.rescored.samples` (252,455,899, real evals) |
| search reference (evaluated positions) | `/mnt/c/TrainingData/evallog/search20ms.samples` (157,140) |
| profile tool (side-by-side %) | `Training/scripts/data_profile.py NAME=FILE ...` |
| duplicates / move counts | `Training/scripts/analyze_dataset.py FILE` |

All samples are 18-byte records (wp, bp, k, color, mlh, result, value), all white to move.
The raw and dedup files have no real evals (value = 1000 placeholder).

**Search reference:** engine option `--log_evals FILE --log_rate N` writes every N-th net
evaluation (18-byte format, one file per process `FILE.<pid>`). Made with 300 games
self-play at 20 ms, `drawbook.book` (balanced), 4 parallel games, rate 1000, **no
tablebases** (Linux matches have none). Run `~/training/bin/python3` for the scripts.

## 1. Dedup shifts the distribution

| Pieces | Raw | Dedup |
|---|---|---|
| 21-24 | 0.75 % | 0.66 % |
| 17-20 | 8.49 % | 8.33 % |
| 13-16 | 17.53 % | 19.69 % |
| 9-12 | 30.74 % | 38.61 % |
| 6-8 | 29.85 % | 29.68 % |
| 2-5 | 12.65 % | **3.03 %** |

Small endgames repeat a lot (small state space), so dedup removes mostly 2-5 piece
positions. Positions with >= 4 kings: 22.9 % raw -> 16.6 % dedup.

Raw data per piece count (total / unique / avg copies):

| Pieces | Raw | Unique | Copies |
|---|---|---|---|
| 2 | 284,888 | 2,821 | 101 |
| 3 | 3,605,784 | 111,646 | 32 |
| 4 | 12,019,646 | 1,730,579 | 7.0 |
| 5 | 28,743,626 | 5,801,162 | 5.0 |
| 6 | 31,597,379 | 18,293,262 | 1.7 |
| 7 | 23,042,181 | 15,558,326 | 1.5 |
| 8 | 51,737,903 | 40,851,309 | 1.3 |
| 2-5 | 44.7M | 7.6M | 5.8 |
| 6-8 | 106.4M | 74.7M | 1.4 |
| 9-12 | 110.9M | 97.3M | 1.14 |
| 13-24 | 95.6M | 72.8M | 1.31 |

Even piece counts are 2-3x as frequent as odd ones (captures come in exchanges).

## 2. Search distribution without tablebases (all evaluated positions)

| Pieces | Search | Dedup |
|---|---|---|
| 21-24 | 0.17 % | 0.60 % |
| 17-20 | 1.70 % | 8.29 % |
| 13-16 | 3.71 % | 19.74 % |
| 9-12 | 7.68 % | 38.32 % |
| 6-8 | 49.34 % | 29.77 % |
| 2-5 | 37.40 % | 3.28 % |

Kings in search: 0: 9.0 %, 1: 3.8 %, 2-3: 21.5 %, >= 4: 65.7 % (dedup: 39 / 17 / 27 / 17 %).
Without tablebases 87 % of evaluations are endgames <= 8 pieces. Caveat: long drawn king
endgames (no adjudication) inflate this.

## 3. Search distribution with 8-piece tablebases (only > 8 pieces reach the net)

This is the CheckerBoard setting (the engine mostly plays with 8-piece tablebases).

| Pieces (> 8 only) | Search | Dedup | Raw |
|---|---|---|---|
| 21-24 | 1.3 % | 1.0 % | 1.3 % |
| 17-20 | 12.8 % | 12.2 % | 14.5 % |
| 13-16 | 27.9 % | 29.3 % | 30.3 % |
| 11-12 | 18.3 % | 24.2 % | 23.0 % |
| 9-10 | 39.6 % | 33.3 % | 30.8 % |

| Kings (> 8 only) | Search | Dedup |
|---|---|---|
| 0 | 58.8 % | 53.2 % |
| 1 | 13.5 % | 18.4 % |
| 2-3 | 15.8 % | 20.8 % |
| >= 4 | 11.9 % | 7.6 % |

Buckets: total variation distance search vs dedup = 0.076. **Above the tablebase range the
dedup piece distribution matches the search well. Decision (user): keep the piece
distribution as it is.** The net must still work without tablebases, so endgames must not
be removed.

## 4. Eval distribution (> 8 pieces)

| abs(eval) | Search | Training (old rescored) |
|---|---|---|
| 0-50 (drawish) | 40.1 % | 40.3 % |
| 51-150 (slight) | 15.3 % | 11.0 % |
| 151-400 (clear) | 15.0 % | 10.0 % |
| 401-1000 (winning) | 29.6 % | 38.4 % |
| > 1000 | 0.0 % | 0.3 % |

Share with **equal material** (king = 1.5 men) inside each band:

| abs(eval) | Search | Training |
|---|---|---|
| 0-50 | 75.8 % | 70.7 % |
| 51-150 | 80.4 % | 76.5 % |
| 151-400 | 50.9 % | 66.1 % |
| 401-1000 | 4.1 % | **22.9 %** |

With a material difference and abs(eval) > 150, the material sign predicts the eval sign in
95-99 % of positions ("easy" positions). The target `sigmoid(eval/127)` saturates for won
positions (eval 600 -> 0.991), so they contribute little to the loss anyway.

Won band split:

| 401-1000 | Search | Training | Weight to match search |
|---|---|---|---|
| with material advantage | 28.4 % | 29.6 % | ~0.96 |
| with equal material | 1.2 % | 8.8 % | ~0.14 |

The excess of won positions in training comes almost entirely from **won positions with
equal material** (shots / combinations, or rescoring artefacts).

## Conclusions / ideas for later

1. Keep the piece distribution of the dedup file (matches the tablebase setting).
2. Under-represented: undecided positions (abs(eval) 51-400: 21 % vs 30 % in search).
3. Over-represented: won positions with equal material (8.8 % vs 1.2 %).
4. Won positions with material advantage: leave at weight ~1 (already saturated, needed
   for calibration).
5. Two ways to change the eval mix:
   - **resampling** (build the file with keep-probabilities per band; to match search
     exactly: 0-50 66 %, 51-150 93 %, 151-400 100 %, 401-1000 51 %, keeps ~2/3 of the data;
     no trainer change),
   - **loss weighting** (weights 1.00 / 1.39 / 1.50 / 0.77; keeps all data, needs a
     per-sample weight in the Rust loader and the loss).
   Recommended first: resampling.
6. Open question: what are the won-with-equal-material positions? Inspect ~20 random
   examples (FEN, stored eval, deep search) before choosing their weight
   (real shots = valuable; artefacts = noise).
7. Further "difficulty" measure: positions where shallow and deep search disagree.
8. Optional: buckets by piece count (existing 12-bucket variant `piece_count_to_bucket`)
   to separate endgame and middlegame weights.
9. Test setup: SPRTs run without tablebases; consider an extra check with tablebase
   adjudication at <= 8 pieces so value-net changes are judged in both settings.
10. Value-net training today: target = search eval only (no game result, by design: the
    reservoir games contain random moves), MSE on sigmoid, validation disabled
    (`limit_val_batches=0`) - turning validation on is a cheap first step.
