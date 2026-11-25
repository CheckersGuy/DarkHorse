# Value net: plan for the 3B-position dataset (2026-11-24)

Goal: gather about **3 billion positions** and train a new value net. Before that, use the
**250M rescored dataset** (the policy data: every sample carries the current engine's root
value) to optimise everything around data, labels, training and architecture, as we are
doing now for the policy net. Related: `data_distribution_findings.md` (distribution
analysis, user decisions: keep the dedup piece distribution, the net must work without
tablebases, no game-result blending because the reservoir games contain random moves).

## 1. Generation cost (the biggest lever)

- Today's rescoring (with child scores, 1 ms + 1 ms, 14-16 workers): ~80-90k positions/min
  -> 3B positions would take **~23-26 days**.
- **Value-only rescoring** (root value, no child searches) should be much faster -> measure.
- **Label budget**: how cheap may labels be (see experiment 2)?
- **Labels during self-play**: the search value of the generating game is free; quality
  depends on the time control of the games.

## 2. Sampling and data quality

- **Reservoir vs Bernoulli sampling**: a 3B reservoir does not fit in RAM (3B x 18 bytes =
  54 GB). Bernoulli sampling (keep each position with probability p, write it to disk,
  external shuffle later) is equivalent for a stream and needs no memory.
- **Correlation within a game**: consecutive positions carry almost the same information.
  Prefer few positions per game (1-5 random plies) from many games (experiment 3).
- **Duplicates**: more repeats at 3B (357M raw -> 252M unique today, mostly endgames).
  Dedup or keep (experiment 4); the dedup piece distribution decision should be rechecked.
- **Game diversity**: opening book, number/placement of random moves (diversity vs
  unrealistic positions), adjudication.
- **Eval mix** (findings): undecided positions (|eval| 51-400) under-represented, won
  positions with equal material over-represented -> resampling by eval band could be
  done directly at generation time (experiment 5).
- **Quiet positions only**: the net only evaluates quiet positions; apply the same filter
  at generation time.

## 3. Labels

- Label noise vs search budget (experiment 2).
- Endgames <= 8 pieces with tablebase values: exact, but the net must work without
  tablebases and TB values (WDL) are on a different scale than search evals -> experiment,
  not a given.
- Loss / scaling: today MSE on sigmoid(eval/127); alternatives cross-entropy, tuned scale K
  (experiment 6).

## 4. Training at 3B scale

- **Speed**: port everything from the policy net to the value training (blocked bucket
  layers, torch.compile, FastRangerLite, lean loop, compact loader with background refill,
  GIL-free Rust fill). Without it one 3B epoch takes days.
- **Storage**: ~54 GB, external global shuffle (like `movegen pshuffle`), on the Linux
  file system rather than /mnt/c.
- **Validation**: so far disabled for the value net (`limit_val_batches=0`). Need a fixed
  validation set + a second one from the search distribution (`--log_evals`), metrics MSE,
  sign accuracy, calibration.
- **Net size**: more data supports a larger net, but the value net already costs ~48 % of
  the engine's time per node -> only worth it if the Elo gain beats the speed loss
  (SPRT at fixed time, nps measured per architecture).

## 5. Architecture experiments (on the 250M set; value net, policy net where it applies)

All architectures must be judged by **validation metrics + engine nps + SPRT at fixed
time**: a better net that is slower can lose Elo. The C++ `Network<L1, L2, L3, Out,
NUM_BUCKETS>` template makes size variants a recompile; new layer types need engine code.

| # | Experiment | Question |
|---|---|---|
| A1 | Accumulator width L1: 1024 / 2048 (now) / 3072 | capacity vs accumulator update + layer_one cost |
| A2 | Hidden layers: 32/32 (now) vs 64/32, 16/32, 32/32/32 | where extra capacity pays off |
| A3 | Buckets: 128 (central-square occupancy, now) vs 12/24 (piece count) vs none | specialisation vs data per bucket |
| A4 | Accumulator activation: pairwise product (now) vs clipped ReLU | quality vs speed (product halves the layer_one input) |
| A5 | Input features: factorised/virtual features (shared piece-square weights during training, folded into the real ones at export) | faster learning, same engine cost |
| A6 | Output: single eval vs WDL head (3 outputs) | calibration, endgame accuracy |
| A7 | Policy net: same width/hidden/bucket questions; sparsity (L1 penalty on the accumulator) for engine speed | quality vs nps |
| A8 | Phase-dependent piece-square inputs: 120 features x 3-4 game phases (e.g. >16, 9-16, <=8 pieces), with factorisation (A5) | phase-specific first layer; accumulator refresh only on phase change (2-3 times per path) -> ~no engine cost |
| **A9** | **Threat inputs (user favourite, 2026-11-24)**, see section 5a | tactics the linear piece-square layer cannot see |

## 5a. Threat inputs (A9) - user favourite

**Idea:** extra input features describing tactical relations, computed from bitboards:
- "piece on square s can be captured by the opponent" (per piece type / colour),
- "piece on square s is backed up / protected" (a capture of it would be answered),
- "pawn on square s has a free path to promotion" (runaway),
- possibly "square s is attacked" / shot patterns (multi-jump threats).

**Why:** the net only evaluates quiet positions (side to move has no capture), but the
*opponent's* threats and shot patterns are exactly what a linear piece-square first layer
struggles to represent. Recent Stockfish versions added threat inputs with a clear Elo gain.

**Incremental updates stay possible:** the engine's update already works by diffing the
feature bitboards of the last evaluated position and the new one (AccumulatorState::apply:
XOR -> changed features -> add/subtract one weight column each). Any feature set that can
be computed from the position as bitboards stays incremental; the cost is proportional to
the number of changed features per evaluation (today ~6). Threat features change for
nearby pieces too, so expect ~8-15 changes per evaluation -> the accumulator update (21 % of
node time today) gets more expensive -> must win an SPRT at fixed time, nps measured.

**Work needed:**
1. Define the threat bitboards (shifts/masks on the 32-square board, both colours, from
   the side-to-move perspective like the existing features).
2. Rust: compute them in the data loader (fill / fill_compact) from the stored position
   (no new data format needed, everything derives from wp/bp/k).
3. Training: input size 120 -> 120 + T, export the extra weight columns.
4. Engine: same bitboards in AccumulatorState::apply (feature extraction + diff), net
   file with the larger accumulator input; bit-exact check vs the Python features.
5. Compare: validation metrics, nps, SPRT at fixed time; start with the cheapest subset
   (e.g. only "capturable" features) and add more if it pays off.

**Roadmap and measured engine cost (2026-11-24).** Cost = additional changed accumulator
features per update in the engine's real search (bench, ACC_STATS block in Network.h; the
same position pairs the engine diffs), value net, piece-square today 6.03:

| Category | Inputs | +changes/update | Update work | Total instructions (est.) |
|---|---|---|---|---|
| 1 capturable (own pawn/king, single jump) | 64 | +0.24 | +4 % | ~+1 % |
| 4 blocked (own/opp x pawn/king, no quiet move) | 128 | +1.13 | +19 % | ~+4 % |
| M3 safe-move thermometer (own/opp >=1,2,3,4,6,8 safe moves) | 12 | +1.44 | +24 % | ~+5 % |
| **1 + 4 + M3 (chosen next, user)** | **204** | **+2.81** | **+47 %** | **~+10 %** |
| M2 safe destination squares (own/opp) | 64 | +7.20 | +119 % | ~+25 % |
| M2 + M3 | 76 | +8.64 | +143 % | ~+30 % (~25 % fewer nps) |

(Accumulator updates are ~21 % of instructions after the speed-ups. Computing the threat /
mobility bitboards themselves comes on top: cheap for 1 and 4, more for M2/M3 because the
quiet moves of both sides are generated and checked for safety.) Policy net: same pattern
(5.81 -> +0.40 cat1, +1.27 cat4, +7.18 M2, +1.46 M3); its share of node time is ~13 %, so the
cost there is ~3x smaller. M2 is probably too expensive for the value net; cheaper
alternatives: own side only, or safe moves per board region.

**Result (2026-11-25): dropped.** Policy-net SPRTs at 20 ms vs A0: T2 (1+4+M3) -7.1 +- 5.7,
T3 (+M2) -7.0 +- 5.7 (both H0) despite better validation (regret 25.44 / 24.77 vs 26.70).
Engine support and training code (`--inputs`) removed again (user). The removed
code is the version before 2026-11-25 (Threats.h, Inputs template parameter, attacked_own,
mobility_white, expand_compact threats=).

**Order (user, 2026-11-24):** v1 (cat. 1) experiment -> cat. 1 + 4 + M3 experiment ->
engine support (opt-in, piece-square-only nets stay supported and the default) -> SPRT.

Also noted, lower priority: draw-counter features for king endgames (moves without capture /
pawn move, in bins); needs the counter in the position and in the training data format.
Not sensible: piece-pair features (the pairwise product activation already models
interactions), "empty square" (linearly dependent), both perspectives concatenated
(the opponent view is just the mirrored position in checkers).

## 6. Small experiments before generating 3B (on the existing 250M)

| # | Experiment | Answers |
|---|---|---|
| 1 | **Data scaling curve**: value nets on 25 / 50 / 100 / 250M, validation + SPRT | Is 3B worth it, or is the curve already flat? **Most important.** |
| 2 | **Label budget**: rescore 10-25M positions with three budgets, same net size | How cheap may labels be -> generation time |
| 3 | **Positions per game**: same number of positions from few vs many games | Sampling rate |
| 4 | **Dedup vs duplicates** at the same data volume | Dedup at 3B? |
| 5 | **Eval-band resampling** (findings) | Data mix |
| 6 | **Loss / scaling** (MSE vs cross-entropy, K) | Training target |
| 7 | **LR, schedule, optimizer, epochs** (as for the policy net now) | Training settings |
| 8 | **Value-only rescoring speed** (positions/min) | Generation time for 3B |
| 9 | **Architecture** (section 5) | Net design for 3B |

## Value training: current state vs policy pipeline (checked 2026-11-25)

Next big block after the threat SPRTs (user). `trainer.py` + `Experimental.Network` +
`BatchDataSet`/`LitDataModule` are still the old path:
- Lightning, `max_epochs=605`, StepLR gamma 0.998 per epoch.
- Bucket layers dense over all 128 buckets + select (no blocked computation).
- No torch.compile, plain RangerLite.
- Rust `BatchProvider.testing` fills full arrays incl. unused legal_moves (B x 128),
  moves, results; GIL probably held; synchronous buffer refill.
- Epoch length hard-coded `100000000 // batch_size`.
- Validation disabled (`limit_val_batches=0`, val path does not exist).
Port: compact value fill (bitboards + eval + bucket, GIL released, background refill,
own eval clamp - the policy fill clamps to +-500), GPU expansion via expand_compact,
blocked bucket layers for the 4096 accumulator + 1-output head, compiled loss,
FastRangerLite, lean loop, validation set + metrics (MSE on sigmoid target, sign accuracy,
calibration per eval band, phase breakdowns), exactness vs the dense path in fp32, export
via save_quantized_bucket must load and evaluate identically in the engine.
**Decided (user, 2026-11-25): the value net trains on the new "root + children" POL1 files
(the rescored policy data), target = the stored root score** (`sample.value` of each
PolicySample, current engine). So the value loader reads POL1 through the same Rust
PolicyDataLoader (background refill, GIL-free, compact fill) and skips the child lists;
it needs its own eval handling (the policy fill clamps the root value to +-500 because it
only uses it as metadata; for the value target the full value incl. mate/TB-like scores
must be passed and converted like the current value training does).
The same files then serve both nets: one data pipeline, one validation set
(reservoir-full.val.children.samples) for policy and value metrics.

**Done (2026-11-25):** `train_value.py` + `Experimental.ValueNetwork` (subclass of
PolicyNetwork: blocked bucket layers, compiled loss, FastRangerLite, lean loop), Rust
`PolicyBatchProvider.fill_value` (bitboards + root value clamped to +-500 as before + bucket,
no children, GIL released), `ThreadedBatchLoader(kind="value")`, GPU `expand_inputs`.
Validation metrics: val/loss (MSE), mae_cp, sign_acc (|eval| >= 25), bias, val_by phase and
eval band (loss, bias, mae_cp). Checks: fill_value == fill_compact; blocked vs dense 3e-8;
identical to the old Network forward/loss; engine (`--value_file`, `--value_dump`) vs Python
corr 0.9994, mean |diff| 6.7 cp (int8 quantisation). Speed with rescoring on the CPU:
old Lightning path 15.9 it/s -> new ~74 it/s (4.7x).
Note: the Rust POL1 loader pops from the end of its buffer, so unshuffled reads come in
reverse file order (harmless).

**QAT (2026-11-25): now the default for both nets, no option (user).** Forward = the engine's
integer arithmetic, bit-exact (value 2.5M positions, policy 16k x 128 outputs, max diff 0).
Value, 100M, 5 epochs, lr 5e-3 step (exp_V_float / exp_V_qat): engine int8 loss 0.01100 vs
0.01098, mae 52.31 vs 52.35 cp, sign acc 96.93 % both; int8 cost float run +0.96 % loss ->
QAT +0.04 %; phase bias spread (engine - float) 5.4 cp -> 0.9 cp; same speed (~72 it/s).
Policy A0 int8 cost (no QAT): regret +0.18, tie-aware top1 -0.06 %, ce_hard +0.004.

**Value rescoring without MLH (user, 2026-11-25):** value labels are the pure search value
(the engine adds the moves-left bonus itself, otherwise it would be counted twice).
Engine `--no_mlh` (USE_MLH, types.h; the eval loop confirms with a first line "no_mlh");
`movegen vrescore <in> <out> <workers> [ms]` (Data::rescoring_value, default 1 ms, POL1 with
empty child lists, read by fill_value); refuses engines without --no_mlh.
**Also for the policy data (user, 2026-11-25, so the 3B set serves both nets):**
`movegen rescore` = root value without MLH, child scores WITH MLH (they separate the
winning moves in decided positions). Engine: `--eval-loop --children --no_mlh` switches
MLH off only around the root search; MLH-off searches use a separate TT key space
(TT_KEY_SALT, set_mlh in types.h) so root and child searches never share entries.
Test 5k positions: root |v| 501-1499 0 %, child scores > 500 still present (MLH).
The running chain (chunks 8-10) still uses the old engine/binary: the 250M set has root
values WITH MLH (fine for the policy net; for value experiments on it, accepted).
Test, 20k positions of chunk 10: |v| 268-500 31.9 %, exactly 500 8.9 %, 501-1500 0 %
(with MLH: 500-1000 37 %). Rough speed (2 workers, CPU shared with the chain): ~660
positions/s per worker -> ~640k/min with 16 workers, 3B in ~3.3 days (experiment 8, to be
measured on an idle CPU). **After the chain is done:** copy the new engine to
Training/Rust/MainEngine and rebuild target/release/movegen (both left untouched while
chunks 8-10 run).

**Root labels: diagnostics, node budget (2026-11-25).** Engine `--eval-loop --label_diag MS`
prints per position the root bookkeeping and a reference search of MS ms (TT cleared).
6k validation positions, root without MLH, reference 50 ms (depth ~18-20):
- 1 ms today = only ~1000-1500 nodes (time checked every 512 nodes), depth 6-7; the search
  practically always stops inside an iteration. Label error vs reference: MAE 29.9 cp.
- Returned value: 62 % last completed iteration, 26 % exact score of the interrupted
  iteration's best move (better than the completed score: 10.5 vs 13.3 cp), 7 % fail-high
  bound (110 cp error, but the completed score is worse: 130), 5 % iteration failing low
  -> old completed score returned (86 cp, bias +23: optimistic).
- "Completed iterations only" is WORSE (MAE 32.2 vs 28.9 at 1500 nodes); keeping the
  partial scores is right. New (labelling only, ROOT_FAIL_LOW_BOUND): failing-low cases
  return the fail-low bound -> loss -1.6 % at 1500 nodes, -6 % at 3000.
- MAE by nodes (750 / 1500 / 3000) and pieces: >=17 68.7/62.6/52.4, 13-16 55.3/50.2/40.0,
  9-12 32.7/27.7/22.5, 7-8 12.7/10.2/8.5, 5-6 7.3/5.9/5.1. More nodes pay off mostly with
  many pieces (endgames are accurate already, although their nps is higher).
- **Node budget instead of time** (`--nodes_phase A,B,C,D` for >=17 / 13-16 / 9-12 / <=8
  pieces, root and child searches): default **2500,2500,1500,750** (Data::DEFAULT_NODES_PHASE),
  same average cost as a uniform 1500 (~today's 1 ms), predicted MAE 27.5 vs 28.9.
  Labels independent of CPU load; identical output for the same input order (the TT
  still carries entries between positions of one worker).
`movegen rescore|vrescore <in> <out> <workers> [A,B,C,D]` use it.

## Suggested order

1. Port the training speed-ups to the value training (prerequisite).
2. Measure 8 (quick).
3. Experiment 1 (scaling curve): decides whether 3B makes sense.
4. Experiments 2 and 3: how to generate the 3B data.
5. Experiments 4-7 and the architecture experiments (section 5) while the GPU is free.
