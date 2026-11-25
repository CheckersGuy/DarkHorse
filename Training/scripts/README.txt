Helper scripts for search testing (not part of the engine build).

bench.sh <out> [--OPT val ...]   depth-18 node counts of cmake-build-debug/MainEngine
                                 on fens40.txt (one number per line)
cmp.sh <ref> <new>               compare two node-count files (identical / total change /
                                 max per-position ratio; > x10 = likely runaway)
ref.txt                          node counts of the current defaults (bench signature 1281642)
sprt.sh <name> OPT=val ...       copy MainEngine to Engines/<name>-dev, start SPRT
                                 (20 ms, unbalanced_70_150.book, [0,3]) detached,
                                 log Training/<name>_sprt_20ms.log
sprt2.sh <name> OPT=val ...      same, stage 2 with [0,2]
spsa_report.py <iter>            top movers of SPSA run 3 (spsa3_log.csv / spsa3_state.json)
spsa_converged.py <iter>         early-stop check used for SPSA run 3 (threshold 0.4 c_end)

Time controls are calibrated for the local machine (see ../bench_reference.txt).
compute_permutation.py  neuron permutation for a .quant net (value: search eval log, policy: POL1 file); output raw size_t[2*L1] for load_permutation; embed it with every new embedded net
