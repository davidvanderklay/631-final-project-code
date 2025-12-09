from leduc_game import LeducGame


def test_game_termination():
    print("Testing Game Termination Logic...")
    game = LeducGame()

    # Test 1: Check-Check Preflop -> Round 2 -> Check-Check -> End
    state = game.get_initial_state()
    print(f"Initial State: {state.history}")

    # Preflop P1 check
    state = state.perform_action("c")
    print(
        f"Action 'c' -> Hist: {state.history}, Round: {state.round_idx}, Finished: {state.finished}"
    )

    # Preflop P2 Check (transition to Round 1)
    state = state.perform_action("c")
    print(
        f"Action 'c' -> Hist: {state.history}, Round: {state.round_idx}, Finished: {state.finished}"
    )

    if state.round_idx != 1:
        print("FAIL: Did not transition to Round 1 after cc")
        return

    # Flop P1 Check
    state = state.perform_action("c")
    print(
        f"Action 'c' -> Hist: {state.history}, Round: {state.round_idx}, Finished: {state.finished}"
    )

    # Flop P2 Check (ends game I think probably)
    state = state.perform_action("c")
    print(
        f"Action 'c' -> Hist: {state.history}, Round: {state.round_idx}, Finished: {state.finished}"
    )

    if not state.finished:
        print("FAIL: Game did not finish after cc/cc")
    else:
        print("SUCCESS: Game finished correctly.")


if __name__ == "__main__":
    test_game_termination()
