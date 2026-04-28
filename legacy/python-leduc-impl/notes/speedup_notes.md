# Speedup Notes — Parallel Self-Play

## Strategy

Parallel self-play with 15 workers using `multiprocessing.Process` and queue-based communication.

### Architecture

- **Workers (15):** Each runs its own copy of the game + agents. Plays batches of 50 episodes,
  sends transitions back to the main process via `mp.Queue`.
- **Main process:** Receives transitions, adds to replay buffers, runs gradient updates (4 per batch),
  and periodically syncs weights back to workers (every 500 episodes).

### Key Design Decisions

1. **Queue-based data transfer:** Workers send raw transitions (numpy arrays), main process handles
   all buffer management and gradient computation. Simple and avoids shared-memory complexity.
2. **Batched weight sync (every 500 eps):** Workers use slightly stale weights. This is standard in
   distributed RL — the staleness is minimal relative to the slow policy drift in NFSP.
3. **Each worker: 1 thread, main: 2 threads.** Total CPU utilization: 17 threads on 15+ core machine.
4. **episodes_per_batch = 50:** Small enough for responsive weight updates, large enough to amortize
   queue overhead.

### Efficient Replay Buffers

- Circular buffer (DQN): list-based with position counter. O(1) add, O(batch) sample.
- Reservoir buffer (average policy): Algorithm R (Vitter 1985). O(1) add, O(batch) sample.
- Both use numpy for batch sampling — avoids per-element tensor creation.

### Evaluation Frequency

- Exploitability computed every 50,000 episodes (tabular on Leduc, ~0.1-0.5 sec per eval).
- Training stats logged every 1,000 episodes.

## Measured Results

*(To be filled after running speedup test)*
