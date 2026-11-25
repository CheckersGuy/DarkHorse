# DarkHorse: project context (as of 2026-11-25)

DarkHorse is an English checkers (8x8) engine: alpha-beta search with Lazy SMP,
NNUE-style nets (value net, policy net for move ordering/LMR/LMP, moves-left net
"MLH"), tablebases on Windows. The current work is a stronger engine through better
training data and nets. Detailed history, SPRT results and all decisions:
`Training/notes/session_notes.md`.

## Working rules (from the repo owner)

- Commit messages / PR texts: only the change description. No co-author, session or
  tool attribution lines, no mention of AI tools. Commit only when asked; the owner
  pushes.
- Report long jobs (rescoring, training, SPRT, SPSA, generation) only at their start
  and end, no intermediate progress unless asked.
- Every new net that gets embedded needs its neuron permutation computed and embedded
  too (`Training/scripts/compute_permutation.py`; check bench is bit-exact).
- Do not kill the owner's own processes. Use they/them for the owner.
- New search features: behind a switch/parameter (default off, bench bit-exact), SPRT
  20 ms `unbalanced_70_150.book` `--sprt 0 3`; H0 -> remove completely, H1 -> default.

## Layout and builds

- Engine: `Checkers/CheckerEngineX/` (main.cpp = CLI, `MainEngine --help` lists every
  option). CMake from the repo root: `cmake . && cmake --build . --target MainEngine
  MainEngine-vnni Train` -> binaries in `cmake-build-debug/`. Bench signature:
  `MainEngine --bench` -> `bench nodes 1281642` (must stay identical for refactors).
- Windows/CheckerBoard: target `Dark` builds `DarkHorse-<ver>-avx2.dll` and
  `DarkHorse-<ver>-vnni.dll` (version in CMakeLists.txt, DARKHORSE_VERSION).
- Match/SPRT tool: `cmake-build-debug/Train --match ...` (`Train --help`); engines are
  taken from `Training/Engines/<name>`, run from `cmake-build-debug/`.
  `Training/scripts/sprt.sh <name> <opt=val>...` copies MainEngine and starts an SPRT.
- Rust data tool + Python loader: `Training/Rust` (nightly Rust, `#![feature]`):
  `cargo build --release --bin movegen`; Python extension `string_sum` (wheel via
  maturin, see `Training/scripts/install_rotate_loader.sh`).
  `movegen` spawns `./MainEngine` from its working directory: copy the CURRENT engine
  build there. NOTE: the committed `Training/Rust/MainEngine` is an older engine
  (23 Nov, used for the 200M rescoring); it lacks `--no_mlh`, node-budget rescoring
  and the reworked generator.
- Training (Python venv ~/training): `Training/train_policy.py`,
  `Training/train_value.py`, model code `Training/Experimental.py`, loop
  `Training/lean_loop.py`. Defaults: QAT (bit-exact engine int8), cosine LR, epoch
  rotation of the loader, CUDA graphs, soft labels T=20 for policy runs.

## Data pipeline

1. **Generate** (self-play, positions sampled from the search trees):
   `movegen generate <book> <out.samples> <workers> <max_samples> [nodes=N] [time=MS]
   [prob=P] [cap=N] [bands=a,b,c,d] [dedup=1] [depth=D] [bloom_fp=P]`
   Defaults reproduce the old generator (1 ms per move, up to 2500 positions per game,
   no filter). Candidate for new data (tested, not yet decided):
   `nodes=1500 prob=0.05 cap=300 dedup=1` (~170 positions per game, ~2x slower per
   position, far less correlated). `bands` = keep factor by |static eval| <=50 /
   51-200 / 201-400 / >400 (opt-in experiment, see data_distribution_findings.md).
   Book: `Training/Positions/tiny.book` (first line is skipped).
2. **Dedup / split / chunks**: `movegen dedup <in> <out>`,
   `movegen split <in> <train> <val> <frac>`, `movegen chunks <in> <prefix> <n>`.
3. **Rescore** (labels, POL1 format = root + children):
   `movegen rescore <in> <out> <workers> [A,B,C,D]` - root value without MLH, child
   scores (policy labels) with MLH; node budget per searched position by pieces
   >=17 / 13-16 / 9-12 / <=8, default 2500,2500,1500,750 (deterministic, no nps
   calibration needed). `movegen vrescore ...` = root value only (value-net data, much
   faster). The engine must print the `no_mlh` banner (new engine only).
   **Cloud / large runs (owner's plan, 2026-11-25):**
   `Training/scripts/cloud_rescore.sh INPUT WORKDIR WORKERS [CHUNK_M=50] [BLOCK_CHUNKS=10]`
   splits the deduplicated input into 50M RANDOM chunks (`movegen chunks ... random`:
   each chunk is a random sample, no dedup drift along the file), rescores them one by
   one and keeps them (a crash loses at most one chunk; rerun the same command to
   resume). Once, after 10 chunks, it builds a merged + shuffled 500M early training
   file `WORKDIR/block.children.samples` (+ `block.ready`) for first checks; at the
   end it builds the full shuffled training file `WORKDIR/train.children.samples`
   (+ `train.ready`). Merging/shuffling happens in the cloud, locally only the finished
   files are needed (3B ~147 GB). `COMPRESS=9` also writes `.zst` files (zstd level 9
   saves ~45 %, level 3 ~40 %, level 19 ~50 % but very slow). Options in the script
   header. Cloud disk for 3B: plan ~600 GB.
4. **Merge + shuffle**: `python3 Training/scripts/merge_policy_files.py OUT IN...`,
   then `movegen pshuffle OUT OUT.shuf`.
5. **Train**: e.g. `train_policy.py --train X --val
   reservoir-full.val.children.samples --l1 1024 --label-mode soft --temperature 20
   --max-epochs 8 --lr 5e-3 --schedule cosine --warmup 0.01 --run NAME`.
   Value net: `train_value.py --train X` (target = stored root value).
6. **Evaluate / SPRT**: `MainEngine --policy_file NET --policy_scale S --policy_eval
   VAL` (offline metrics incl. pmax/entropy), then SPRT vs the embedded net
   (`--opts1 policy_file=/abs/NET POLICY_SCALE_X100=225`).

Data files (local machine): `/mnt/c/TrainingData/`; current training set
`reservoir-full.train.cum8.children.samples` (199.9M positions, shuffled),
validation `reservoir-full.val.children.samples` (always the same file).

## Cloud machine notes (many cores)

- Memory per process: rescoring worker ~100 MB (64 MB hash), generation worker ~50 MB,
  match engine ~55 MB (2 per parallel game). Plan RAM accordingly.
- `movegen` and `Train` raise the open-file limit themselves (fixed 2026-11-25; before,
  `Train` hung with >~120 parallel games under a 1024 limit).
- Time-based matches on SMT machines search fewer nodes per move: run
  `Training/bench_parallel.sh` on both machines and scale `--time`. Rescoring and
  generation with node budgets are machine-independent.
- Hard-coded paths in scripts (`/mnt/c/TrainingData`, `/home/leaguerobin/...`) must be
  adapted.
- Bloom filters are sized per job (since 2026-11-25): `generate` uses
  max_samples x 1.1 entries at `bloom_fp` (default 1e-3, ~1.8 bytes per entry:
  3e9 positions ~6 GB); `movegen dedup <in> <out> [expected] [fp]` sizes for the input
  file (18 bytes per sample) at fp 1e-4 (~2.4 bytes per entry: 3e9 ~7.2 GB). Both
  print their filter size at the start.

## Current state (2026-11-25)

- Policy net trained on 200M positions (`Training/Networks/exp_D200_cos_8ep_199.9M_7.quant`,
  8 epochs): int8 val regret 25.60 / tie-aware top1 0.8596, better than 100M with 8
  epochs (26.58) and 100M with 150 epochs (26.25). More data beats more epochs.
  SPRT vs the embedded net (20 ms [0,3], scale 225) stopped by the owner after 9786
  games: Elo +1.6 +- 2.5, LLR 0.11, no decision (100M net: +0.9 +- 2.9). Not embedded.
- Rescoring of chunks 9-10 was dropped (owner); the next data comes from the reworked
  generator.
- Value net: training pipeline ported and verified, MLH-free value labels decided; plan
  and experiment list in `Training/value_net_plan.md`, data analysis in
  `Training/data_distribution_findings.md`.

## Next steps (owner's plan)

1. Post-training plan for the policy net: 1-parameter POLICY_SCALE SPSA,
   full SPSA run 4 with POLICY_SCALE_X100 fixed, verification SPRT.
2. Experiments on the 200M set before generating a lot more data (value net: data
   scaling, positions per game, label budget, eval-band resampling).
3. Generate new data on a cloud machine with the reworked generator (settings from
   step 2), rescore with the new engine and node budget.
