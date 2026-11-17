"""
Q-Learning Traffic Signal Optimization
=======================================
A Tabular Q-Learning reinforcement learning agent that learns to control
traffic signals at a 4-way intersection to minimize total vehicle waiting time.

The agent observes discretized queue levels on each road and the current green
phase, then decides which road should receive the next green signal.

Usage:
    python QLearning.py                    # Train and evaluate
    python QLearning.py --episodes 5000    # Custom episode count
    python QLearning.py --demo-only        # Load saved Q-table, evaluate only
    python QLearning.py --verbose          # Show detailed training progress

Author: AI-Based Traffic Management System Project
"""

import numpy as np
import argparse
import os
import sys

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

NUM_ROADS = 4
QUEUE_BINS = 4          # Discretization levels: empty(0), low(1), medium(2), high(3)
QUEUE_BIN_EDGES = [0, 3, 8, 15]  # Thresholds: 0=empty, 1-3=low, 4-8=med, 9-15=high, 16+=high

# Vehicle priority weights (preserved from original project)
VEHICLE_PRIORITY = {'bike': 1, 'car': 2, 'truck_bus': 3}

# Environment parameters
GREEN_DURATION = 30         # seconds per green phase
VEHICLES_CLEARED_PER_GREEN = 5  # max vehicles cleared during one green phase
MAX_QUEUE_SIZE = 20         # cap on vehicles queued per road
STARVATION_THRESHOLD = 120  # seconds before starvation penalty kicks in
SWITCH_PENALTY = 2          # penalty for switching to a different road

# Arrival rates (Poisson lambda per road per step) — highly asymmetric to simulate
# realistic traffic where some roads are much busier than others.
# Total avg arrivals (~7.5/step) slightly exceeds clearing rate (5/step),
# forcing the agent to learn prioritization rather than cycling blindly.
DEFAULT_ARRIVAL_RATES = [3.0, 1.0, 2.5, 1.0]  # Roads 0-3

# Q-Learning hyperparameters (defaults, overridable via CLI)
DEFAULT_EPISODES = 2000
DEFAULT_STEPS_PER_EPISODE = 200
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_EXPLORATION_START = 1.0
DEFAULT_EXPLORATION_MIN = 0.01
DEFAULT_EXPLORATION_DECAY = 0.997

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q_TABLE_PATH = os.path.join(SCRIPT_DIR, "q_table.npy")


# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC INTERSECTION ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class TrafficIntersectionEnv:
    """
    Simulates a 4-way traffic intersection environment for RL training.

    State Space:
        (queue_level_road0, queue_level_road1, queue_level_road2, queue_level_road3, current_green)
        Each queue_level ∈ {0, 1, 2, 3} (empty/low/medium/high)
        current_green ∈ {0, 1, 2, 3} (which road has the green signal)
        Total states = 4^4 × 4 = 1,024

    Action Space:
        4 actions — give green to road 0, 1, 2, or 3.

    Reward:
        Negative total waiting time across all roads, with penalties for
        signal switching and starvation.
    """

    def __init__(self, arrival_rates=None):
        self.arrival_rates = arrival_rates or DEFAULT_ARRIVAL_RATES
        self.num_roads = NUM_ROADS
        self.reset()

    def reset(self):
        """Reset the environment to initial conditions."""
        # Vehicle queues: actual count of waiting vehicles per road
        self.queues = np.random.randint(0, 8, size=self.num_roads).astype(float)
        # Waiting time accumulated per road (in seconds)
        self.waiting_times = np.zeros(self.num_roads)
        # Current green signal
        self.current_green = np.random.randint(0, self.num_roads)
        # Step counter
        self.step_count = 0
        return self._get_state()

    def _discretize_queue(self, queue_size):
        """Convert raw vehicle count to a discrete bin index (0-3)."""
        queue_size = int(queue_size)
        if queue_size <= QUEUE_BIN_EDGES[0]:
            return 0  # empty
        elif queue_size <= QUEUE_BIN_EDGES[1]:
            return 1  # low
        elif queue_size <= QUEUE_BIN_EDGES[2]:
            return 2  # medium
        else:
            return 3  # high

    def _get_state(self):
        """Encode the current environment state as a single integer index."""
        bins = [self._discretize_queue(q) for q in self.queues]
        # Encode as a single integer: b0 * 256 + b1 * 64 + b2 * 16 + b3 * 4 + green
        state = 0
        for b in bins:
            state = state * QUEUE_BINS + b
        state = state * self.num_roads + self.current_green
        return state

    @staticmethod
    def state_space_size():
        """Total number of possible states."""
        return (QUEUE_BINS ** NUM_ROADS) * NUM_ROADS  # 4^4 * 4 = 1024

    @staticmethod
    def action_space_size():
        """Total number of possible actions."""
        return NUM_ROADS  # 4

    def step(self, action, arrivals=None):
        """
        Execute one time step in the environment.

        Args:
            action (int): Road index (0-3) to assign green signal.
            arrivals (array-like, optional): Pre-generated arrival counts per road.
                If None, arrivals are sampled from Poisson distribution.
                Used for fair evaluation comparisons.

        Returns:
            next_state (int): New state index after the step.
            reward (float): Reward received for this step.
            info (dict): Diagnostic information for logging.
        """
        self.step_count += 1

        # ── 1. Apply signal switching penalty ──
        switch_cost = 0
        if action != self.current_green:
            switch_cost = SWITCH_PENALTY  # Cost of yellow phase transition
        self.current_green = action

        # ── 2. Clear vehicles from the green road ──
        pre_clear_queue = self.queues[action]  # Track queue before clearing
        cleared = min(self.queues[action], VEHICLES_CLEARED_PER_GREEN)
        self.queues[action] = max(0, self.queues[action] - cleared)

        # ── 3. New vehicle arrivals ──
        if arrivals is not None:
            # Use pre-generated arrivals for deterministic evaluation
            for i in range(self.num_roads):
                self.queues[i] = min(self.queues[i] + arrivals[i], MAX_QUEUE_SIZE)
        else:
            # Sample from Poisson distribution (used during training)
            for i in range(self.num_roads):
                arr = np.random.poisson(self.arrival_rates[i])
                self.queues[i] = min(self.queues[i] + arr, MAX_QUEUE_SIZE)

        # ── 4. Update waiting times ──
        for i in range(self.num_roads):
            if i == action:
                self.waiting_times[i] = 0  # Green road's waiting time resets
            else:
                # Waiting time increases proportional to queue size
                self.waiting_times[i] += GREEN_DURATION

        # ── 5. Calculate reward ──
        total_waiting = np.sum(self.waiting_times)
        total_queue = np.sum(self.queues)

        # Starvation penalty: any road waiting too long
        starvation_penalty = 0
        for i in range(self.num_roads):
            if self.waiting_times[i] > STARVATION_THRESHOLD:
                starvation_penalty += (self.waiting_times[i] - STARVATION_THRESHOLD) * 2

        # Base cost: penalize total waiting time and queue buildup
        reward = -(total_waiting / 100.0) - (total_queue / 10.0) - switch_cost - starvation_penalty / 100.0

        # ── 6. Smart clearing bonus ──
        # Strongly reward clearing busy roads — this is what differentiates
        # the RL agent from round-robin. Pre-clear queue is used so the agent
        # gets credit for choosing a congested road.
        clearing_bonus = cleared * 0.5  # Proportional to vehicles actually cleared
        # Extra bonus if the agent chose the road with the highest queue
        max_queue_before = max(pre_clear_queue, *[self.queues[j] for j in range(self.num_roads) if j != action])
        if pre_clear_queue >= max_queue_before:
            clearing_bonus += 2.0  # Bonus for targeting the worst road
        reward += clearing_bonus

        next_state = self._get_state()

        info = {
            'queues': self.queues.copy(),
            'waiting_times': self.waiting_times.copy(),
            'cleared': cleared,
            'action': action,
            'total_waiting': total_waiting,
            'total_queue': total_queue,
            'starvation_penalty': starvation_penalty,
            'switch_cost': switch_cost,
        }

        return next_state, reward, info

    def get_queue_summary(self):
        """Return human-readable queue summary."""
        labels = ['empty', 'low', 'medium', 'high']
        summary = {}
        for i in range(self.num_roads):
            level = self._discretize_queue(self.queues[i])
            summary[f'Road_{i+1}'] = {
                'vehicles': int(self.queues[i]),
                'level': labels[level],
                'waiting_time': int(self.waiting_times[i]),
            }
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.

    The agent maintains a Q-table mapping (state, action) pairs to expected
    future rewards, and updates it using the Bellman equation:

        Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(self, state_size, action_size, learning_rate=DEFAULT_LEARNING_RATE,
                 discount_factor=DEFAULT_DISCOUNT_FACTOR,
                 exploration_rate=DEFAULT_EXPLORATION_START,
                 exploration_min=DEFAULT_EXPLORATION_MIN,
                 exploration_decay=DEFAULT_EXPLORATION_DECAY):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = exploration_min
        self.epsilon_decay = exploration_decay

        # Initialize Q-table with small random values to break ties
        self.q_table = np.random.uniform(low=-0.01, high=0.01,
                                          size=(state_size, action_size))

    def choose_action(self, state):
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        """Update Q-table using the Bellman equation."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return td_error

    def decay_exploration(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path=Q_TABLE_PATH):
        """Save trained Q-table to disk."""
        np.save(path, self.q_table)
        print(f"  Q-table saved to: {path}")

    def load(self, path=Q_TABLE_PATH):
        """Load a pre-trained Q-table from disk."""
        if not os.path.exists(path):
            print(f"  Error: No saved Q-table found at {path}")
            return False
        self.q_table = np.load(path)
        self.epsilon = 0.0  # No exploration when using saved model
        print(f"  Q-table loaded from: {path}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(episodes=DEFAULT_EPISODES, steps_per_episode=DEFAULT_STEPS_PER_EPISODE,
          verbose=False):
    """
    Train the Q-Learning agent on the traffic intersection environment.

    Args:
        episodes: Number of training episodes.
        steps_per_episode: Steps per episode.
        verbose: If True, print per-episode details.

    Returns:
        agent: The trained QLearningAgent.
        reward_history: List of total rewards per episode.
    """
    env = TrafficIntersectionEnv()
    agent = QLearningAgent(
        state_size=TrafficIntersectionEnv.state_space_size(),
        action_size=TrafficIntersectionEnv.action_space_size(),
    )

    reward_history = []
    best_reward = float('-inf')

    print("=" * 65)
    print("  Q-LEARNING TRAFFIC SIGNAL OPTIMIZATION — TRAINING")
    print("=" * 65)
    print(f"  Episodes:        {episodes}")
    print(f"  Steps/Episode:   {steps_per_episode}")
    print(f"  State Space:     {agent.state_size} states")
    print(f"  Action Space:    {agent.action_size} actions")
    print(f"  Learning Rate:   {agent.lr}")
    print(f"  Discount Factor: {agent.gamma}")
    print(f"  Exploration:     {agent.epsilon:.2f} → {agent.epsilon_min}")
    print("=" * 65)
    print()

    log_interval = max(1, episodes // 20)  # Print ~20 progress lines

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_waiting = 0

        for step in range(steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            total_waiting += info['total_waiting']

        agent.decay_exploration()
        avg_waiting = total_waiting / steps_per_episode
        reward_history.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward

        if verbose or (episode + 1) % log_interval == 0 or episode == 0:
            print(f"  Episode {episode+1:>5}/{episodes}  |  "
                  f"Reward: {total_reward:>8.1f}  |  "
                  f"Avg Wait: {avg_waiting:>7.1f}s  |  "
                  f"ε: {agent.epsilon:.4f}  |  "
                  f"Best: {best_reward:>8.1f}")

    print()
    print("  Training complete!")
    print(f"  Final exploration rate: {agent.epsilon:.4f}")
    print(f"  Best episode reward:   {best_reward:.1f}")

    # Show convergence: compare first 10% vs last 10%
    window = max(1, episodes // 10)
    early_avg = np.mean(reward_history[:window])
    late_avg = np.mean(reward_history[-window:])
    improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
    print(f"  Avg reward (first {window} eps):  {early_avg:.1f}")
    print(f"  Avg reward (last {window} eps):   {late_avg:.1f}")
    print(f"  Improvement:                 {improvement:+.1f}%")
    print()

    agent.save()
    return agent, reward_history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, num_steps=50):
    """
    Run the trained agent in evaluation mode and display step-by-step decisions.

    Also runs a round-robin baseline for comparison.

    Args:
        agent: Trained QLearningAgent (epsilon should be 0 or near-0).
        num_steps: Number of evaluation steps.
    """
    print()
    print("=" * 65)
    print("  EVALUATION — TRAINED AGENT vs ROUND-ROBIN BASELINE")
    print("=" * 65)

    # Pre-generate arrival patterns so BOTH strategies face identical traffic
    np.random.seed(12345)
    initial_queues = np.random.randint(0, 8, size=NUM_ROADS).astype(float)
    initial_green = np.random.randint(0, NUM_ROADS)
    arrival_schedule = []
    for _ in range(num_steps):
        step_arrivals = [np.random.poisson(DEFAULT_ARRIVAL_RATES[i]) for i in range(NUM_ROADS)]
        arrival_schedule.append(step_arrivals)

    # ── Evaluate Trained Agent ──
    env = TrafficIntersectionEnv()
    env.queues = initial_queues.copy()
    env.current_green = initial_green
    env.waiting_times = np.zeros(NUM_ROADS)
    env.step_count = 0
    state = env._get_state()

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    total_reward_agent = 0
    total_waiting_agent = 0
    total_cleared_agent = 0
    total_queue_agent = 0

    print()
    print("  +-------------------------------------------------------------+")
    print("  |                  TRAINED AGENT DECISIONS                     |")
    print("  +-------+----------------------------+--------+---------------+")
    print("  | Step  | Queues [R1, R2, R3, R4]    | Action | Reward        |")
    print("  +-------+----------------------------+--------+---------------+")

    for step in range(num_steps):
        action = agent.choose_action(state)
        next_state, reward, info = env.step(action, arrivals=arrival_schedule[step])
        state = next_state
        total_reward_agent += reward
        total_waiting_agent += info['total_waiting']
        total_cleared_agent += info['cleared']
        total_queue_agent += info['total_queue']

        q = info['queues'].astype(int)
        queue_str = f"[{q[0]:>2}, {q[1]:>2}, {q[2]:>2}, {q[3]:>2}]"
        action_str = f"-> R{action+1} [G]"

        if step < 20 or step >= num_steps - 5:  # Show first 20 and last 5 steps
            print(f"  | {step+1:>4}  | {queue_str:<26} | {action_str:<6} | {reward:>+11.2f}  |")
        elif step == 20:
            print(f"  |  ...  | {'...':^26} | {'...':^6} | {'...':^13} |")

    print("  +-------+----------------------------+--------+---------------+")
    avg_waiting_agent = total_waiting_agent / num_steps
    avg_queue_agent = total_queue_agent / num_steps

    agent.epsilon = old_epsilon

    # ── Evaluate Round-Robin Baseline with IDENTICAL traffic ──
    env_baseline = TrafficIntersectionEnv()
    env_baseline.queues = initial_queues.copy()
    env_baseline.current_green = initial_green
    env_baseline.waiting_times = np.zeros(NUM_ROADS)
    env_baseline.step_count = 0

    total_reward_baseline = 0
    total_waiting_baseline = 0
    total_cleared_baseline = 0
    total_queue_baseline = 0

    for step in range(num_steps):
        action = step % NUM_ROADS  # Simple round-robin
        _, reward, info = env_baseline.step(action, arrivals=arrival_schedule[step])
        total_reward_baseline += reward
        total_waiting_baseline += info['total_waiting']
        total_cleared_baseline += info['cleared']
        total_queue_baseline += info['total_queue']

    avg_waiting_baseline = total_waiting_baseline / num_steps
    avg_queue_baseline = total_queue_baseline / num_steps

    # -- Comparison --
    print()
    print("  +-------------------------------------------------------------+")
    print("  |              PERFORMANCE COMPARISON                         |")
    print("  +-----------------------+--------------+---------------------+")
    print("  | Metric                | Q-Learning   | Round-Robin         |")
    print("  +-----------------------+--------------+---------------------+")
    print(f"  | Total Reward          | {total_reward_agent:>+10.1f}  | {total_reward_baseline:>+17.1f}  |")
    print(f"  | Avg Waiting Time (s)  | {avg_waiting_agent:>10.1f}  | {avg_waiting_baseline:>17.1f}  |")
    print(f"  | Vehicles Cleared      | {total_cleared_agent:>10.0f}  | {total_cleared_baseline:>17.0f}  |")
    print(f"  | Avg Queue Size        | {avg_queue_agent:>10.1f}  | {avg_queue_baseline:>17.1f}  |")
    print("  +-----------------------+--------------+---------------------+")

    if total_reward_agent > total_reward_baseline:
        pct = ((total_reward_agent - total_reward_baseline) / abs(total_reward_baseline)) * 100
        print(f"\n  [OK] Trained agent outperforms round-robin by {pct:.1f}%")
    else:
        print(f"\n  [!!] Round-robin performed better. Consider more training episodes.")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Q-TABLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_q_table(agent):
    """Print a summary analysis of the learned Q-table."""
    print("  +-------------------------------------------------------------+")
    print("  |                  Q-TABLE ANALYSIS                          |")
    print("  +-------------------------------------------------------------+")

    q = agent.q_table

    # How many states have been meaningfully visited (non-trivial Q-values)
    visited_mask = np.any(np.abs(q) > 0.1, axis=1)
    visited_count = np.sum(visited_mask)
    print(f"  States visited (|Q| > 0.1):  {visited_count} / {agent.state_size}")

    # Action distribution: how often each action is preferred
    preferred_actions = np.argmax(q, axis=1)
    print(f"  Preferred action distribution:")
    for a in range(agent.action_size):
        count = np.sum(preferred_actions == a)
        pct = count / agent.state_size * 100
        bar = '#' * int(pct / 2)
        print(f"    Road {a+1}: {count:>4} states ({pct:5.1f}%)  {bar}")

    # Average Q-values per action
    print(f"  Average Q-value per action:")
    for a in range(agent.action_size):
        avg_q = np.mean(q[:, a])
        print(f"    Road {a+1}: {avg_q:>+8.3f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q-Learning Traffic Signal Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python QLearning.py                    Train with defaults (2000 episodes)
  python QLearning.py --episodes 5000    Train with custom episode count
  python QLearning.py --demo-only        Load saved Q-table, evaluate only
  python QLearning.py --verbose          Show per-episode training details
        """
    )
    parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES,
                        help=f'Number of training episodes (default: {DEFAULT_EPISODES})')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS_PER_EPISODE,
                        help=f'Steps per episode (default: {DEFAULT_STEPS_PER_EPISODE})')
    parser.add_argument('--demo-only', action='store_true',
                        help='Skip training, load saved Q-table and evaluate')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed per-episode training output')
    parser.add_argument('--eval-steps', type=int, default=50,
                        help='Number of evaluation steps (default: 50)')

    args = parser.parse_args()

    print()
    print("  +===============================================================+")
    print("  |     AI-BASED TRAFFIC MANAGEMENT SYSTEM                       |")
    print("  |     Q-Learning Signal Optimization Module                    |")
    print("  +===============================================================+")
    print()

    if args.demo_only:
        # Load and evaluate only
        agent = QLearningAgent(
            state_size=TrafficIntersectionEnv.state_space_size(),
            action_size=TrafficIntersectionEnv.action_space_size(),
        )
        if not agent.load():
            print("  Cannot run demo without a saved Q-table. Train first:")
            print("    python QLearning.py")
            sys.exit(1)
        analyze_q_table(agent)
        evaluate(agent, num_steps=args.eval_steps)
    else:
        # Train and evaluate
        agent, reward_history = train(
            episodes=args.episodes,
            steps_per_episode=args.steps,
            verbose=args.verbose,
        )
        analyze_q_table(agent)
        evaluate(agent, num_steps=args.eval_steps)


if __name__ == "__main__":
    main()
