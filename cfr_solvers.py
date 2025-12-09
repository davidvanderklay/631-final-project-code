import random
from collections import defaultdict
from leduc_game import LeducGame, LeducState, RANK_CHARS


class CFRNode:
    """
    Stores Regret Sum and Strategy Sum for specific Information Set.
    """

    __slots__ = ("regret_sum", "strategy_sum", "possible_actions", "n_actions")

    def __init__(self, actions):
        self.possible_actions = actions
        self.n_actions = len(actions)
        self.regret_sum = {a: 0.0 for a in actions}
        self.strategy_sum = {a: 0.0 for a in actions}

    def get_strategy(self, use_positive_regret_only=False):
        """
        Computes current strategy with Regret Matching.
        """
        strategy = {}
        positive_regret_sum = 0.0

        for a in self.possible_actions:
            r = self.regret_sum[a]
            # Normal RM floors negative regrets to 0 for calculation
            r = max(0, r)
            strategy[a] = r
            positive_regret_sum += r

        if positive_regret_sum > 0:
            for a in self.possible_actions:
                strategy[a] /= positive_regret_sum
        else:
            # Default uniform
            uniform = 1.0 / self.n_actions
            for a in self.possible_actions:
                strategy[a] = uniform

        return strategy

    def get_average_strategy(self):
        """
        Returns accumulated average strategy.
        """
        avg_strategy = {}
        total_sum = sum(self.strategy_sum.values())

        if total_sum > 0:
            for a in self.possible_actions:
                avg_strategy[a] = self.strategy_sum[a] / total_sum
        else:
            uniform = 1.0 / self.n_actions
            for a in self.possible_actions:
                avg_strategy[a] = uniform

        return avg_strategy


class CFRSolver:
    """
    Abstract Base Class for CFR Solvers.
    """

    def __init__(self, game):
        self.game = game
        self.nodes = {}  # InfoSet_String -> CFRNode map
        self.iterations = 0

    def train(self, iterations):
        for _ in range(iterations):
            self.iterations += 1
            # Player 0 traverse
            self._cfr_iteration(self.game.get_initial_state(), 1.0, 1.0, 0)
            # Player 1 traverse
            self._cfr_iteration(self.game.get_initial_state(), 1.0, 1.0, 1)

    def get_average_strategy_profile(self):
        """
        Returns full strategy profile for Exploitability calculation.
        Dict: InfoSet -> {Action: Prob}
        """
        profile = {}
        for info_set, node in self.nodes.items():
            profile[info_set] = node.get_average_strategy()
        return profile

    def _get_node(self, info_set, legal_actions):
        if info_set not in self.nodes:
            self.nodes[info_set] = CFRNode(legal_actions)
        return self.nodes[info_set]

    def _cfr_iteration(self, state, p0_prob, p1_prob, traversal_player):
        """
        Recursive CFR step. IMPLEMENT FOR SUBCLASSES.
        """
        raise NotImplementedError("Base class cannot run CFR.")


class VanillaCFR(CFRSolver):
    def _cfr_iteration(self, state, p0_prob, p1_prob, traversal_player):
        if state.is_terminal():
            payoff = state.get_payoff()
            return payoff if traversal_player == 0 else -payoff

        current_player = state.current_player

        # Opponent Turn recursion
        if current_player != traversal_player:
            info_set = state.get_information_set()
            legal_actions = state.get_legal_actions()
            node = self._get_node(info_set, legal_actions)

            strategy = node.get_strategy()

            total_val = 0.0
            for action in legal_actions:
                prob = strategy.get(action, 0.0)
                if prob == 0:
                    continue

                next_state = state.perform_action(action)

                if current_player == 0:
                    val = self._cfr_iteration(
                        next_state, p0_prob * prob, p1_prob, traversal_player
                    )
                else:
                    val = self._cfr_iteration(
                        next_state, p0_prob, p1_prob * prob, traversal_player
                    )

                total_val += val * prob
            return total_val

        # Traversal Player recursion
        info_set = state.get_information_set()
        legal_actions = state.get_legal_actions()
        node = self._get_node(info_set, legal_actions)

        strategy = node.get_strategy()

        action_utils = {}
        node_util = 0.0

        for action in legal_actions:
            next_state = state.perform_action(action)

            if current_player == 0:
                val = self._cfr_iteration(
                    next_state, p0_prob, p1_prob, traversal_player
                )
            else:
                val = self._cfr_iteration(
                    next_state, p0_prob, p1_prob, traversal_player
                )

            action_utils[action] = val
            node_util += strategy[action] * val

        # Regret and Strategy Accumulation
        my_prob = p0_prob if current_player == 0 else p1_prob
        opp_prob = p1_prob if current_player == 0 else p0_prob

        for action in legal_actions:
            regret = action_utils[action] - node_util
            node.regret_sum[action] += regret * opp_prob
            node.strategy_sum[action] += strategy[action] * my_prob

        return node_util


class CFRPlus(CFRSolver):
    """
    CFR+:
    1. Regret Matching
    2. Linear Averaging
    """

    def _cfr_iteration(self, state, p0_prob, p1_prob, traversal_player):
        if state.is_terminal():
            payoff = state.get_payoff()
            return payoff if traversal_player == 0 else -payoff

        current_player = state.current_player
        info_set = state.get_information_set()
        legal_actions = state.get_legal_actions()
        node = self._get_node(info_set, legal_actions)

        # CFR+: Get strategy
        strategy = node.get_strategy()

        if current_player != traversal_player:
            total_val = 0.0
            for action in legal_actions:
                prob = strategy.get(action, 0.0)
                next_state = state.perform_action(action)
                if current_player == 0:
                    val = self._cfr_iteration(
                        next_state, p0_prob * prob, p1_prob, traversal_player
                    )
                else:
                    val = self._cfr_iteration(
                        next_state, p0_prob, p1_prob * prob, traversal_player
                    )
                total_val += val * prob
            return total_val

        # Our turn
        action_utils = {}
        node_util = 0.0

        for action in legal_actions:
            next_state = state.perform_action(action)
            if current_player == 0:
                val = self._cfr_iteration(
                    next_state, p0_prob, p1_prob, traversal_player
                )
            else:
                val = self._cfr_iteration(
                    next_state, p0_prob, p1_prob, traversal_player
                )
            action_utils[action] = val
            node_util += strategy[action] * val

        # Update steps
        opp_prob = p1_prob if current_player == 0 else p0_prob
        my_prob = p0_prob if current_player == 0 else p1_prob

        for action in legal_actions:
            regret = action_utils[action] - node_util

            # CFR+: Regret sum update - floor at 0
            node.regret_sum[action] = max(
                0, node.regret_sum[action] + (regret * opp_prob)
            )

            # CFR+: Linear averaging for strategy
            weight = max(1, self.iterations)
            node.strategy_sum[action] += strategy[action] * my_prob * weight

        return node_util


class PruningCFR(VanillaCFR):
    """
    Regret-Based Pruning.
    """

    def __init__(self, game, prune_threshold=-2000, prune_duration=10):
        super().__init__(game)
        self.prune_threshold = prune_threshold
        self.prune_duration = prune_duration
        self.pruning_flags = {}

    def _cfr_iteration(self, state, p0_prob, p1_prob, traversal_player):
        if not state.is_terminal() and state.current_player == traversal_player:
            info_set = state.get_information_set()
            legal_actions = state.get_legal_actions()
            node = self._get_node(info_set, legal_actions)

            # Filter actions to explore
            actions_to_explore = []
            for action in legal_actions:
                key = (info_set, action)
                if key in self.pruning_flags:
                    if self.pruning_flags[key] > 0:
                        self.pruning_flags[key] -= 1
                        continue
                    else:
                        del self.pruning_flags[key]
                actions_to_explore.append(action)

            if not actions_to_explore:
                actions_to_explore = legal_actions

            strategy = node.get_strategy()
            action_utils = {}
            node_util = 0.0

            for action in legal_actions:
                if action in actions_to_explore:
                    next_state = state.perform_action(action)
                    if state.current_player == 0:
                        val = self._cfr_iteration(
                            next_state, p0_prob, p1_prob, traversal_player
                        )
                    else:
                        val = self._cfr_iteration(
                            next_state, p0_prob, p1_prob, traversal_player
                        )
                    action_utils[action] = val
                else:
                    action_utils[action] = 0.0

                node_util += strategy[action] * action_utils[action]

            opp_prob = p1_prob if state.current_player == 0 else p0_prob
            my_prob = p0_prob if state.current_player == 0 else p1_prob

            for action in legal_actions:
                if action in actions_to_explore:
                    regret = action_utils[action] - node_util
                    node.regret_sum[action] += regret * opp_prob

                    if node.regret_sum[action] < self.prune_threshold:
                        self.pruning_flags[(info_set, action)] = self.prune_duration

                node.strategy_sum[action] += strategy[action] * my_prob

            return node_util

        # If not our turn or terminal, fallback to Vanilla logic
        if state.is_terminal():
            payoff = state.get_payoff()
            return payoff if traversal_player == 0 else -payoff

        info_set = state.get_information_set()
        legal_actions = state.get_legal_actions()
        node = self._get_node(info_set, legal_actions)
        strategy = node.get_strategy()
        total_val = 0.0
        for action in legal_actions:
            prob = strategy.get(action, 0.0)
            if prob == 0:
                continue
            next_state = state.perform_action(action)
            if state.current_player == 0:
                val = self._cfr_iteration(
                    next_state, p0_prob * prob, p1_prob, traversal_player
                )
            else:
                val = self._cfr_iteration(
                    next_state, p0_prob, p1_prob * prob, traversal_player
                )
            total_val += val * prob
        return total_val


if __name__ == "__main__":
    from exploitability import ExploitabilityCalculator

    game = LeducGame(rng_seed=42)
    calc = ExploitabilityCalculator(game)

    print("--- Testing Vanilla CFR (100 Iterations) ---")
    solver = VanillaCFR(game)
    solver.train(100)
    profile = solver.get_average_strategy_profile()
    exp = calc.compute_exploitability(profile)
    print(f"Vanilla CFR Exploitability: {exp:.5f}")

    print("\n--- Testing CFR+ (100 Iterations) ---")
    solver_plus = CFRPlus(game)
    solver_plus.train(100)
    profile_plus = solver_plus.get_average_strategy_profile()
    exp_plus = calc.compute_exploitability(profile_plus)
    print(f"CFR+ Exploitability: {exp_plus:.5f}")

    print("\n--- Testing Pruning CFR (100 Iterations) ---")
    solver_prune = PruningCFR(game, prune_threshold=-5.0, prune_duration=5)
    solver_prune.train(100)
    profile_prune = solver_prune.get_average_strategy_profile()
    exp_prune = calc.compute_exploitability(profile_prune)
    print(f"Pruning CFR Exploitability: {exp_prune:.5f}")
