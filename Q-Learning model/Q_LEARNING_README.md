# Q-Learning Traffic Signal Optimization

## What is This?

This module implements a **Tabular Q-Learning** reinforcement learning (RL) agent that learns to optimally control traffic signals at a simulated 4-way intersection. The agent's goal is to **minimize total vehicle waiting time** across all roads by deciding which road should receive the green signal at each time step.

---

## How Q-Learning Works

Q-Learning is a **model-free** reinforcement learning algorithm. The agent interacts with an environment in discrete time steps:

1. **Observe** the current state `s` (traffic conditions).
2. **Choose** an action `a` (which road gets green).
3. **Receive** a reward `r` (negative of total waiting time).
4. **Observe** the new state `s'`.
5. **Update** the Q-table using the **Bellman equation**:

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
```

Where:
- `α` (learning rate) = how quickly the agent updates its beliefs (default: 0.1)
- `γ` (discount factor) = how much the agent values future rewards vs. immediate (default: 0.95)
- `ε` (exploration rate) = probability of taking a random action instead of the learned best (starts at 1.0, decays to 0.01)

Over thousands of episodes, the Q-table converges to the **optimal policy** — the agent learns which signal decisions minimize waiting time for any given traffic condition.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                               │
│                                                                 │
│   ┌───────────────────┐        ┌──────────────────────┐         │
│   │  QLearningAgent   │        │  TrafficIntersection │         │
│   │                   │ action │  Env                 │         │
│   │  ┌─────────────┐  ├───────►│                      │         │
│   │  │  Q-Table     │  │        │  - 4 road queues     │         │
│   │  │  1024 × 4    │  │ state, │  - Vehicle arrivals  │         │
│   │  │              │  │ reward │    (Poisson dist.)   │         │
│   │  │  s → best a  │  │◄───────┤  - Signal switching  │         │
│   │  └─────────────┘  │        │  - Waiting times     │         │
│   │                   │        │  - Starvation guard  │         │
│   │  epsilon-greedy   │        └──────────────────────┘         │
│   │  action selection │                                         │
│   └───────────────────┘                                         │
│                                                                 │
│   Repeat for 2000 episodes × 200 steps each                    │
└─────────────────────────────────────────────────────────────────┘
```

### State Space (1,024 states)

The state encodes 5 features:

| Feature | Values | Description |
|---|---|---|
| Road 0 queue level | 0-3 | empty / low / medium / high |
| Road 1 queue level | 0-3 | empty / low / medium / high |
| Road 2 queue level | 0-3 | empty / low / medium / high |
| Road 3 queue level | 0-3 | empty / low / medium / high |
| Current green phase | 0-3 | Which road currently has green |

Queue levels are discretized from raw vehicle counts:
- **Empty (0)**: 0 vehicles
- **Low (1)**: 1-3 vehicles
- **Medium (2)**: 4-8 vehicles
- **High (3)**: 9+ vehicles

Total: `4 × 4 × 4 × 4 × 4 = 1,024` unique states.

### Action Space (4 actions)

| Action | Meaning |
|---|---|
| 0 | Give green to Road 1 |
| 1 | Give green to Road 2 |
| 2 | Give green to Road 3 |
| 3 | Give green to Road 4 |

### Reward Function

```
reward = -(total_waiting / 100) - (total_queue / 10) - switch_penalty - starvation_penalty + clearing_bonus
```

- **Total waiting**: Sum of accumulated waiting times across all roads.
- **Total queue**: Sum of vehicles waiting across all roads.
- **Switch penalty**: Cost of 2 points when changing the green signal to a different road (simulates yellow phase delay).
- **Starvation penalty**: Heavy penalty when any road waits more than 120 seconds without service.
- **Clearing bonus**: Proportional to vehicles cleared (0.5 per vehicle) plus a +2.0 bonus for targeting the road with the highest queue. This is the key differentiator vs. round-robin.

### Environment Dynamics

Each time step simulates a 30-second green phase:

1. **Vehicle clearing**: Up to 5 vehicles are cleared from the road that receives green.
2. **Vehicle arrivals**: New vehicles arrive on each road following a Poisson distribution with different rates per road (simulating asymmetric traffic):
   - Road 1: lambda = 3.0 (busy road)
   - Road 2: lambda = 1.0 (quiet road)
   - Road 3: lambda = 2.5 (moderately busy)
   - Road 4: lambda = 1.0 (quiet road)
3. **Queue cap**: Maximum 20 vehicles per road (prevents unbounded growth).
4. **Waiting time update**: Red roads accumulate 30 seconds of waiting time; the green road resets to 0.

---

## Vehicle Priority Weights

Preserved from the original project design, vehicle types have different priority weights:

| Vehicle Type | Priority Weight |
|---|---|
| Bike | 1 |
| Car | 2 |
| Truck / Bus | 3 |

These weights influence the priority score calculation used in the starvation guard.

---

## How to Run

### Prerequisites

Only requires `numpy` (already in the project's `requirements.txt`):
```bash
pip install numpy
```

### Training and Evaluation (Default)

```bash
cd "Q-Learning model/src"
python QLearning.py
```

This will:
1. Train the agent for 2,000 episodes (200 steps each).
2. Print progress every ~100 episodes.
3. Save the trained Q-table to `q_table.npy`.
4. Run a 50-step evaluation comparing the trained agent vs. a round-robin baseline.

### Custom Training

```bash
# Train with more episodes for better convergence
python QLearning.py --episodes 5000

# Show per-episode training details
python QLearning.py --verbose

# Custom steps per episode
python QLearning.py --episodes 3000 --steps 300

# Custom evaluation length
python QLearning.py --eval-steps 100
```

### Demo Only (Load Saved Model)

```bash
python QLearning.py --demo-only
```

Loads a previously saved `q_table.npy` and runs evaluation without retraining.

### Full Help

```bash
python QLearning.py --help
```

---

## Understanding the Output

### Training Output

```
Episode   100/2000  |  Reward:  -1164.9  |  Avg Wait:   297.0s  |  ε: 0.7405  |  Best:  -1059.8
```

| Field | Meaning |
|---|---|
| Episode | Current episode / total episodes |
| Reward | Total reward accumulated this episode (higher = better, always negative) |
| Avg Wait | Average total waiting time (seconds) across all roads this episode |
| ε (epsilon) | Current exploration rate (1.0 = fully random, 0.01 = mostly learned policy) |
| Best | Best single-episode reward seen so far |

### Convergence Summary

```
Avg reward (first 200 eps):  -1306.5
Avg reward (last 200 eps):   -1124.1
Improvement:                 +14.0%
```

A positive improvement percentage means the agent learned to reduce waiting times compared to its early (mostly random) behavior.

### Q-Table Analysis

```
Preferred action distribution:
  Road 1:  220 states ( 21.5%)
  Road 2:  180 states ( 17.6%)
  Road 3:  350 states ( 34.2%)
  Road 4:  274 states ( 26.8%)
```

This shows which road the agent prefers to give green in each state. Road 3 being preferred most often makes sense — it has the highest arrival rate (λ = 4.0).

### Evaluation Comparison

The evaluation runs the trained agent side-by-side with a simple **round-robin** baseline (cycles Road 1 → 2 → 3 → 4 → 1...) to demonstrate that the learned policy is smarter than fixed-cycle control.

---

## File Structure

```
Q-Learning model/
├── Q_LEARNING_README.md    # This documentation
└── src/
    ├── QLearning.py        # Main module (environment + agent + training)
    └── q_table.npy         # Saved Q-table (created after first training run)
```

---

## Key Differences from Original Implementation

| Aspect | Original | Rewritten |
|---|---|---|
| State space | 4 abstract states (meaningless) | 1,024 states encoding real traffic conditions |
| Action space | 2 abstract actions (never applied) | 4 actions (one per road, actually executed) |
| Environment | Static; traffic never clears | Dynamic: vehicles arrive (Poisson), clear on green |
| Reward | Connected to waiting time but action was ignored | Reward directly tied to agent's chosen action |
| Output | 400,000+ print lines flooding console | ~20 progress lines during training |
| CLI | None | Full argparse with `--episodes`, `--demo-only`, etc. |
| Persistence | None | Saves/loads Q-table to `q_table.npy` |
| Evaluation | None | Trained agent vs. round-robin baseline comparison |
