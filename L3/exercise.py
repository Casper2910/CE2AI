import random
from statistics import mean, pstdev

gamma = 0.9


def step(state, action):
    """Return (next_state, reward, done)."""
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


def policy_probs(state):
    """Return dict: action -> prob."""
    if state == "A":
        return {0: 0.7, 1: 0.3}

    if state == "B":
        return {0: 0.4, 1: 0.6}

    return {0: 1.0}


def sample_action(probs):
    x = random.random()
    cum = 0.0
    for a, p in probs.items():
        cum += p
        if x <= cum:
            return a
    return a  # fallback


def run_episode(max_steps=100):
    s = "A"
    rewards = []
    traj = []

    for _ in range(max_steps):
        a = sample_action(policy_probs(s))
        s_next, r, done = step(s, a)

        traj.append((s, a, r, s_next, done))
        rewards.append(r)

        s = s_next
        if done:
            break

    G = sum((gamma ** t) * r for t, r in enumerate(rewards))
    return G, traj


def main():
    N = 2000
    returns = []

    for _ in range(N):
        G, _ = run_episode()
        returns.append(G)

    print("Mean return:", mean(returns))
    print("Std dev (population):", pstdev(returns))


if __name__ == "__main__":
    main()