import random
from cfr_solvers import CFRSolver, CFRNode
from leduc_game import LeducGame, LeducState


class MCCFRSolver(CFRSolver):
    """
    Base class for Monte Carlo CFR variants.
    Handles RNG seeding and averaging modes.
    """

    def __init__(self, game, seed=None, linear_averaging=False):
        super().__init__(game)
        self.rng = random.Random(seed)
        self.linear_averaging = linear_averaging
        # Control the game deck shuffling explicitly
        self.game_rng = random.Random(seed)

    def _get_strategy_weight(self):
        # 1.0 for Uniform, 't' for Linear
        return self.iterations if self.linear_averaging else 1.0


class ExternalSamplingMCCFR(MCCFRSolver):
    """
    External Sampling MCCFR:
    - Traversal Player: Branches on ALL actions
    - Opponent/Chance: Samples one action
    - Low variance, higher cost per iteration than Outcome Sampling.
    """

    def train(self, iterations):
        for _ in range(iterations):
            self.iterations += 1

            # Update Player 0
            state = self._get_shuffled_state()
            self._walk_tree(state, 0)

            # Update Player 1
            state = self._get_shuffled_state()
            self._walk_tree(state, 1)

    def _get_shuffled_state(self):
        # Manually shuffle deck using seeded RNG
        deck = [0, 1, 2, 3, 4, 5]
        self.game_rng.shuffle(deck)
        cards = (deck[0], deck[1], deck[2])
        return LeducState(cards=cards)

    def _walk_tree(self, state, traversal_player):
        if state.is_terminal():
            payoff = state.get_payoff()
            return payoff if traversal_player == 0 else -payoff

        current_player = state.current_player
        info_set = state.get_information_set()
        legal_actions = state.get_legal_actions()
        node = self._get_node(info_set, legal_actions)
        strategy = node.get_strategy()

        if current_player == traversal_player:
            # Traversal Player: Iterate all actions
            # Calculate expected value node by summing weighted child values.

            action_values = {}
            node_value = 0.0

            for action in legal_actions:
                next_state = state.perform_action(action)
                val = self._walk_tree(next_state, traversal_player)
                action_values[action] = val
                node_value += strategy[action] * val

            # Update Regrets
            for action in legal_actions:
                regret = action_values[action] - node_value
                node.regret_sum[action] += regret

                node.strategy_sum[action] += (
                    strategy[action] * self._get_strategy_weight()
                )

            return node_value

        else:
            # --- Opponent: Sample ONE action ---
            # Sample based on current strategy profile

            cumulative = 0.0
            r = self.rng.random()
            selected_action = legal_actions[0]

            for action in legal_actions:
                cumulative += strategy[action]
                if r <= cumulative:
                    selected_action = action
                    break

            next_state = state.perform_action(selected_action)
            return self._walk_tree(next_state, traversal_player)


class OutcomeSamplingMCCFR(MCCFRSolver):
    """
    Outcome Sampling MCCFR:
    - Samples a single trajectory through the tree (One branch everywhere).
    - High variance, very fast per iteration.
    - Uses importance sampling corrections (epsilon-greedy).
    """

    def __init__(self, game, seed=None, linear_averaging=False, epsilon=0.6):
        super().__init__(game, seed, linear_averaging)
        self.epsilon = epsilon  # Exploration parameter

    def train(self, iterations):
        for _ in range(iterations):
            self.iterations += 1

            for p in [0, 1]:
                state = self._get_shuffled_state()
                self._walk_tree(state, p, 1.0, 1.0)

    def _get_shuffled_state(self):
        deck = [0, 1, 2, 3, 4, 5]
        self.game_rng.shuffle(deck)
        cards = (deck[0], deck[1], deck[2])
        return LeducState(cards=cards)

    def _walk_tree(self, state, traversal_player, s_prob, p_tail):
        """
        s_prob: Probability of reaching history based on sampling policy epsilon included
        p_tail: Probability of the remainder of the trajectory based on player policy.
        """
        if state.is_terminal():
            payoff = state.get_payoff()
            return (payoff if traversal_player == 0 else -payoff) / s_prob

        current_player = state.current_player
        info_set = state.get_information_set()
        legal_actions = state.get_legal_actions()
        node = self._get_node(info_set, legal_actions)
        strategy = node.get_strategy()

        # 1. Define sampling policy
        # To ensure we visit all branches, mix strategy with uniform random.
        n_actions = len(legal_actions)
        epsilon_share = self.epsilon / n_actions

        sampling_probs = {}
        for a in legal_actions:
            sampling_probs[a] = (1 - self.epsilon) * strategy[a] + epsilon_share

        # 2. Sample Action
        r = self.rng.random()
        cumulative = 0.0
        action = legal_actions[0]
        for a in legal_actions:
            cumulative += sampling_probs[a]
            if r <= cumulative:
                action = a
                break

        # 3. Recurse
        new_s_prob = s_prob * sampling_probs[action]

        next_state = state.perform_action(action)
        utility = self._walk_tree(next_state, traversal_player, new_s_prob, p_tail)

        # 4. Update if traversal player
        if current_player == traversal_player:
            w = utility * s_prob  # recovers raw payoff scaled by tails

            for a in legal_actions:
                val_a = 0.0
                if a == action:
                    val_a = utility

                val_policy = strategy[action] * utility
                regret = val_a - val_policy

                node.regret_sum[a] += regret * self._get_strategy_weight()

                # Strategy accumulation
                node.strategy_sum[a] += (
                    strategy[a] * self._get_strategy_weight()
                )  # * (1/s_prob)?

        return utility


if __name__ == "__main__":
    from exploitability import ExploitabilityCalculator

    game = LeducGame(rng_seed=99)
    calc = ExploitabilityCalculator(game)

    print("--- Testing External Sampling MCCFR (1000 Iterations) ---")
    # ES needs more iterations than Vanilla but is faster per iter
    solver_es = ExternalSamplingMCCFR(game, seed=42)
    solver_es.train(1000)
    prof_es = solver_es.get_average_strategy_profile()
    print(f"ES MCCFR Exploitability: {calc.compute_exploitability(prof_es):.5f}")

    print("\n--- Testing Outcome Sampling MCCFR (10000 Iterations) ---")
    # OS needs MANY more iterations because of high variance
    solver_os = OutcomeSamplingMCCFR(game, seed=42, epsilon=0.5)
    solver_os.train(10000)
    prof_os = solver_os.get_average_strategy_profile()
    print(f"OS MCCFR Exploitability: {calc.compute_exploitability(prof_os):.5f}")
