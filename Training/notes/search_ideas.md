# Search ideas from the code review (2026-11-28), not yet implemented

Already implemented and in testing on 2026-11-28 (see session_notes / SPRT logs):
TT 40-bit signature + checksum against torn entries (ttA), 64-byte TT clusters (ttB),
SE_SINGLE_FIX, SE_EXCL_TT, PC_EXCL_FIX, LMP1_COUNT (depth-1 nodes without the policy
net), TT_CAPTURES (TT moves for captures via the move-list index), CheckerBoard
win/loss only for proven results, dead code removed.

Constants are first guesses.

## Speedups (test with non-regression SPRTs [-3, 1])
1. Policy net output layer: compute only the logits of the legal moves (~8 of 128 rows).
2. `has_jumps()` for both sides once per node (now up to 5 calls per node).
3. `get_moves`: the jumper mask is computed twice (has_jumps + loop_captures).
4. Duplicate king captures (same from/to/captures via different paths): deduplicate.
5. Minor: TT age has 6 bits and wraps after 64 searches within one game; a forced root
   move returns a bare qsearch score (display / draw adjudication streak).

## New checkers-specific features (normal SPRTs [0, 3])
6. Threat check deeper in qsearch: the 1-ply threat search (opponent has a capture,
   we have none) only runs at qsearch depth 0; allow it down to depth -2
   (QS_THREAT_PLIES = 2), only if the threatening side would win material.
7. Forcing quiet moves (the opponent must capture next, `creates_shot`): exempt them
   from LMP; second variant also 1 ply less LMR (FORCING_LMR_FP = 1024).
8. Endgame safety: below 8 pieces, LMR -0.5 ply (ENDGAME_PIECES = 8,
   ENDGAME_LMR_FP = 512).
9. Razoring blend, like the accepted RFP blend: return (W*eval + (1024-W)*alpha)/1024,
   RAZOR_BLEND_W = 512.
10. Killer move (1 per ply) right after the TT move when the policy's best probability
    is < 0.5 (KILLER_MAX_P_X100 = 50).

## Refined versions of failed ideas
11. #13 shot moves in qsearch (-4.9): only real 2-for-1 shots (we give one piece and
    capture at least two back), only at qsearch depth 0, only with >= 8 pieces.
12. #17 promotion extension (-3.2 in bundle A): 0.5 ply (fractional depth exists now),
    only if the new king cannot be captured immediately.
13. #37 smother / #38 safe-move gating (both failed): "two-reply" extension of 0.4 ply
    (TWO_REPLY_EXT_FP = 400) for positions with exactly two legal moves.
14. #36 ETC (-5.2): the cost was the problem; only at depth >= 8 and for the first 3
    moves (ETC_MIN_DEPTH = 8, ETC_MOVES = 3).
15. Correction history (failed earlier): key = hash of the men only (both colours) +
    side to move, 16384 entries, update (search - static) * depth / 8 clamped to
    +-64, applied with weight 1/4 to the static eval for pruning decisions only.
16. Singular extensions inside singular verification subtrees: today is_sing_search
    disables SE in the whole subtree; block it only at the node itself.

## Constants that are not tunable yet (candidates for the next SPSA)
- SE_MIN_DEPTH (6), singular search depth (depth-1)/2, SE_MARGIN as an integer 1..6
  (one step = +-50 %; use SE_MARGIN_X10)
- ProbCut TT gate `info.depth >= depth - 4`, shallow TT cutoff `depth - 1`
- LMP: first move index 2, depth limit 8 (array size); LMR gates depth >= 2 and move
  index >= 1; exponent of the probability -> LMR index mapping (1/p)^gamma
- fail-high blend (best*depth + K*beta)/(depth + K), K = 1
- aspiration: 4 re-searches before the full window, fail-low midpoint (alpha+beta)/2
- capture ordering 14 (king) / 10 (man); qsearch threat search depth 1
- draw score +-1 (contempt), moves-left scale 300 and ramp top (MAX_EVAL), eval
  clamp 500
- time management constants (off in fixed-time matches; matter in CheckerBoard)
- Lazy SMP depth-skipping tables (SkipSize / SkipPhase)
