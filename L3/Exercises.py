import random
from statistics import mean, pstdev


# Shared MDP definition

gamma = 0.9
states = ["A", "B"]
actions = [0, 1]


def step(state: str, action: int):
    # Deterministic environment step function.
    # Returns:
    #    next_state (str)
    #    reward (float)
    #    done (bool)
    if state == "A":
        if action == 0:
            return "B", 1.0, False
        else:
            return "T", 0.0, True

    if state == "B":
        if action == 0:
            return "A", 0.0, False
        else:
            return "T", 2.0, True

    return "T", 0.0, True

# Question 1 — Monte Carlo simulation

def policy_probs(state: str) -> dict[int, float]:
    # Stochastic policy π(a|s).
    if state == "A":
        return {0: 0.7, 1: 0.3}
    if state == "B":
        return {0: 0.4, 1: 0.6}
    return {0: 1.0}


def sample_action(probs: dict[int, float]) -> int:
    # Sample action according to probability dictionary.
    x = random.random()
    cumulative = 0.0
    for a, p in probs.items():
        cumulative += p
        if x <= cumulative:
            return a
    return a  # fallback


def run_episode(max_steps: int = 100):
    # Run one episode starting from state A.
    # Returns:
    #    G0 (discounted return)
    #    trajectory (list of transitions)
    s = "A"
    rewards = []
    trajectory = []

    for _ in range(max_steps):
        a = sample_action(policy_probs(s))
        s_next, r, done = step(s, a)

        trajectory.append((s, a, r, s_next, done))
        rewards.append(r)

        s = s_next
        if done:
            break

    G0 = sum((gamma ** t) * r for t, r in enumerate(rewards))
    return G0, trajectory


def monte_carlo_simulation(N: int = 5000):
    # Run multiple episodes and report summary statistics.
    returns = []
    for _ in range(N):
        G, _ = run_episode()
        returns.append(G)

    print("=== Question 1: Monte Carlo ===")
    print(f"Episodes: {N}")
    print("Mean return:", mean(returns))
    print("Std dev (population):", pstdev(returns))
    print()

    return mean(returns)

# Question 2 — Policy Evaluation (Iterative)

def transition_reward(state: str, action: int):
    # Return (next_state, reward) without done flag.
    if state == "A":
        return ("B", 1.0) if action == 0 else ("T", 0.0)
    if state == "B":
        return ("A", 0.0) if action == 0 else ("T", 2.0)
    return ("T", 0.0)


def pi(state: str) -> dict[int, float]:
    # Same stochastic policy.
    return policy_probs(state)


def policy_evaluation(tol: float = 1e-12, max_iters: int = 10000):
    # Iterative policy evaluation using Bellman expectation backup.
    V = {"A": 0.0, "B": 0.0, "T": 0.0}

    for _ in range(max_iters):
        delta = 0.0

        for s in ["A", "B"]:
            v_old = V[s]
            v_new = 0.0

            for a, p in pi(s).items():
                s_next, r = transition_reward(s, a)
                v_new += p * (r + gamma * V[s_next])

            V[s] = v_new
            delta = max(delta, abs(v_new - v_old))

        if delta < tol:
            break

    return V

# Question 3 — Bellman Optimality Backup

def step_det(state: str, action: int):
    # Deterministic transition (same as step without done).
    return transition_reward(state, action)


def max_q_next(Q: dict, s_next: str):
    # Compute max_a Q(s_next, a).
    if s_next == "T":
        return 0.0
    return max(Q[(s_next, a)] for a in actions)


def bellman_opt_backup(Q: dict, s: str, a: int):
    # One Bellman optimality backup.
    s_next, r = step_det(s, a)
    return r + gamma * max_q_next(Q, s_next)


def epsilon_greedy(Q: dict, s: str, eps: float = 0.1):
    # ε-greedy action selection.
    if random.random() < eps:
        return random.choice(actions)

    vals = [Q[(s, a)] for a in actions]
    m = max(vals)
    best = [a for a in actions if Q[(s, a)] == m]
    return random.choice(best)


def greedy_action(Q: dict, s: str):
    # Greedy action with random tie-breaking.
    vals = [Q[(s, a)] for a in actions]
    m = max(vals)
    best = [a for a in actions if Q[(s, a)] == m]
    return random.choice(best)


def run_sweeps(K: int = 6):
    # Run synchronous Bellman optimality sweeps.
    Q = {(s, a): 0.0 for s in states for a in actions}

    print("Question 3: Bellman Optimality Sweeps ")

    for k in range(1, K + 1):
        Q_new = Q.copy()

        for s in states:
            for a in actions:
                Q_new[(s, a)] = bellman_opt_backup(Q, s, a)

        Q = Q_new

        print(f"Sweep {k}:")
        print("  Greedy(A) =", greedy_action(Q, "A"),
              "| Greedy(B) =", greedy_action(Q, "B"))
        print("  Q(A,0), Q(A,1) =", Q[("A", 0)], Q[("A", 1)])
        print("  Q(B,0), Q(B,1) =", Q[("B", 0)], Q[("B", 1)])
        print()

    return Q

# Main


if __name__ == "__main__":

    # Q1
    mc_estimate = monte_carlo_simulation(N=5000)

    # Q2
    V = policy_evaluation()
    print("Question 2: Policy Evaluation")
    print("V^pi(A) =", V["A"])
    print("V^pi(B) =", V["B"])
    print("Monte Carlo estimate (A) ≈", mc_estimate)
    print()

    # Q3
    run_sweeps(K=6)