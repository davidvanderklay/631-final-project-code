import random

# Dedfine constants
RANK_CHARS = {0: "J", 1: "Q", 2: "K"}
SUIT_CHARS = {0: "s", 1: "h"}
ACTIONS = ["f", "c", "r"]  # Fold, Check/Call, Raise


class LeducState:
    __slots__ = (
        "cards",
        "history",
        "round_idx",
        "current_player",
        "bets",
        "pot",
        "finished",
    )

    def __init__(
        self,
        cards,
        history="",
        round_idx=0,
        current_player=0,
        bets=None,
        pot=2,
        finished=False,
    ):
        self.cards = cards
        self.history = history
        self.round_idx = round_idx
        self.current_player = current_player
        self.bets = bets if bets is not None else [1, 1]
        self.pot = pot
        self.finished = finished

    def is_terminal(self):
        return self.finished

    def get_legal_actions(self):
        """
        Returns a list of legal actions ['f', 'c', 'r'].
        """
        if self.finished:
            return []

        actions = ["c", "r"]

        # Can fold if opponent bet more
        if self.bets[1 - self.current_player] > self.bets[self.current_player]:
            actions.append("f")

        # Check raise limit
        round_history = (
            self.history.split("/")[self.round_idx]
            if "/" in self.history
            else self.history
        )
        if round_history.count("r") >= 2:
            if "r" in actions:
                actions.remove("r")

        return actions

    def perform_action(self, action):
        """
        Returns new LeducState object after the action is taken.
        """
        new_history = self.history + action
        new_bets = list(self.bets)
        new_pot = self.pot
        new_round_idx = self.round_idx
        new_player = 1 - self.current_player
        is_finished = False

        opponent = 1 - self.current_player
        raise_amount = 2 if self.round_idx == 0 else 4

        if action == "f":
            is_finished = True

        elif action == "c":
            amount_to_call = new_bets[opponent] - new_bets[self.current_player]
            if amount_to_call > 0:
                new_bets[self.current_player] += amount_to_call
                new_pot += amount_to_call

            round_str = new_history.split("/")[-1]

            # If bets match, check if proceed
            if new_bets[0] == new_bets[1]:
                if len(round_str) >= 2:
                    if self.round_idx == 0:
                        new_round_idx = 1
                        new_history += "/"
                        new_player = 0
                    else:
                        is_finished = True

        elif action == "r":
            amount_to_add = (
                new_bets[opponent] - new_bets[self.current_player]
            ) + raise_amount
            new_bets[self.current_player] += amount_to_add
            new_pot += amount_to_add

        return LeducState(
            cards=self.cards,
            history=new_history,
            round_idx=new_round_idx,
            current_player=new_player,
            bets=new_bets,
            pot=new_pot,
            finished=is_finished,
        )

    def get_information_set(self):
        """
        Format: [PrivateCard]:[History] or [PrivateCard]:[History]:[BoardCard]
        """
        p_card_rank = self.cards[self.current_player] // 2
        rank_char = RANK_CHARS[p_card_rank]

        if self.round_idx == 0:
            return f"{rank_char}:{self.history}"
        else:
            board_rank = self.cards[2] // 2
            board_char = RANK_CHARS[board_rank]
            return f"{rank_char}:{self.history}:{board_char}"

    def get_payoff(self):
        if not self.finished:
            raise ValueError("Game not finished")

        # Check fold
        last_action = self.history[-1]
        if last_action == "f":
            winner = self.current_player
            loser = 1 - winner
            return self.bets[loser] if winner == 0 else -self.bets[loser]

        # Showdown
        rank0 = self.cards[0] // 2
        rank1 = self.cards[1] // 2
        board = self.cards[2] // 2

        is_pair0 = rank0 == board
        is_pair1 = rank1 == board

        winner = -1
        if is_pair0 and not is_pair1:
            winner = 0
        elif is_pair1 and not is_pair0:
            winner = 1
        elif is_pair0 and is_pair1:
            winner = -1
        else:
            if rank0 > rank1:
                winner = 0
            elif rank1 > rank0:
                winner = 1
            else:
                winner = -1

        if winner == -1:
            return 0
        return self.bets[1] if winner == 0 else -self.bets[0]


class LeducGame:
    def __init__(self, rng_seed=None):
        self.deck = [0, 1, 2, 3, 4, 5]
        if rng_seed is not None:
            random.seed(rng_seed)

    def get_initial_state(self):
        random.shuffle(self.deck)
        cards = (self.deck[0], self.deck[1], self.deck[2])
        return LeducState(cards=cards)


if __name__ == "__main__":
    # Quick test
    game = LeducGame(rng_seed=42)
    state = game.get_initial_state()

    print("--- Starting Random Simulation ---")
    while not state.is_terminal():
        legal = state.get_legal_actions()
        action = random.choice(legal)
        print(
            f"P{state.current_player} does {action} -> Hist: {state.history + action}"
        )
        state = state.perform_action(action)

    print(f"\nGame Over. History: {state.history}")
    print(f"Payoff (P0 perspective): {state.get_payoff()}")
