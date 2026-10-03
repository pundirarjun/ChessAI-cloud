from dataclasses import dataclass

import numpy as np
import chess

from environment.state_encoder import StateEncoder
from environment.action_encoder import ActionEncoder

from mcts.node import Node
from mcts.mcts import MCTS


# ==========================================================
# TRAINING SAMPLE
# ==========================================================

@dataclass
class TrainingSample:

    state: np.ndarray
    policy: np.ndarray
    player: int


# ==========================================================
# SELF-PLAY RESULT
# ==========================================================

@dataclass
class SelfPlayResult:

    training_data: list
    result: int | None
    termination: str
    moves_played: int
    completed: bool


# ==========================================================
# SELF-PLAY GAME
# ==========================================================

class SelfPlayGame:

    def __init__(self):

        self.samples = []

    # ======================================================
    # STORE POSITION
    # ======================================================

    def add_position(
        self,
        board,
        policy
    ):

        state = StateEncoder.encode(
            board
        )

        player = (
            1
            if board.turn == chess.WHITE
            else -1
        )

        sample = TrainingSample(

            state=state,

            policy=np.array(
                policy,
                dtype=np.float32
            ),

            player=player
        )

        self.samples.append(
            sample
        )

    # ======================================================
    # CREATE TRAINING DATA
    # ======================================================

    def get_training_data(
        self,
        result
    ):

        training_data = []

        for sample in self.samples:

            # --------------------------------------------------
            # Draw
            # --------------------------------------------------

            if result == 0:

                value = 0.0

            # --------------------------------------------------
            # White won
            # --------------------------------------------------

            elif result == 1:

                value = float(
                    sample.player
                )

            # --------------------------------------------------
            # Black won
            # --------------------------------------------------

            else:

                value = float(
                    -sample.player
                )

            training_data.append(
                (
                    sample.state,
                    sample.policy,
                    value
                )
            )

        return training_data


# ==========================================================
# FINALIZE ONE GAME
# ==========================================================

def _finalize_game(
    board,
    game
):

    outcome = board.outcome(
        claim_draw=True
    )

    if outcome is None:

        return SelfPlayResult(

            training_data=[],
            result=None,
            termination="UNKNOWN",
            moves_played=len(game.samples),
            completed=False

        )

    # ------------------------------------------------------
    # Determine result
    # ------------------------------------------------------

    if outcome.winner is None:

        result = 0

    elif outcome.winner == chess.WHITE:

        result = 1

    else:

        result = -1

    termination = str(
        outcome.termination
    )

    training_data = (
        game.get_training_data(
            result
        )
    )

    return SelfPlayResult(

        training_data=training_data,
        result=result,
        termination=termination,
        moves_played=len(game.samples),
        completed=True

    )


# ==========================================================
# PLAY MULTIPLE SELF-PLAY GAMES
# ==========================================================
#
# This is the GPU-efficient self-play path.
#
# Several independent games are advanced together.
# Their MCTS neural-network evaluations are combined
# into the same GPU batches.
#
# ==========================================================

def play_games(

    model,

    num_games=8,

    num_simulations=100,

    max_moves=200,

    temperature=1.0,

    temperature_moves=20,

    late_temperature=0.0,

    dirichlet_alpha=0.3,

    dirichlet_epsilon=0.25,

    batch_size=128

):

    if num_games <= 0:

        return []

    # ======================================================
    # SHARED OBJECTS
    # ======================================================

    action_encoder = ActionEncoder()

    mcts = MCTS(

        model=model,

        action_encoder=action_encoder

    )

    # ======================================================
    # CREATE GAMES
    # ======================================================

    boards = [
        chess.Board()
        for _ in range(num_games)
    ]

    games = [
        SelfPlayGame()
        for _ in range(num_games)
    ]

    move_numbers = [
        1
        for _ in range(num_games)
    ]

    results = [
        None
        for _ in range(num_games)
    ]

    # ======================================================
    # ACTIVE GAME INDICES
    # ======================================================

    active_indices = list(
        range(num_games)
    )

    # ======================================================
    # SELF-PLAY LOOP
    # ======================================================

    while active_indices:

        # --------------------------------------------------
        # Remove games that are already terminal.
        # --------------------------------------------------

        still_active = []

        for index in active_indices:

            board = boards[index]

            if board.is_game_over(
                claim_draw=True
            ):

                if results[index] is None:

                    results[index] = (
                        _finalize_game(
                            board,
                            games[index]
                        )
                    )

            else:

                still_active.append(
                    index
                )

        active_indices = still_active

        if not active_indices:
            break

        # ==================================================
        # CHECK MAX-MOVE LIMIT
        # ==================================================

        still_active = []

        for index in active_indices:

            if move_numbers[index] > max_moves:

                print(
                    f"Game {index + 1}: "
                    "MAX_MOVES reached."
                )

                results[index] = SelfPlayResult(

                    training_data=[],
                    result=None,
                    termination="MAX_MOVES",
                    moves_played=len(
                        games[index].samples
                    ),
                    completed=False

                )

            else:

                still_active.append(
                    index
                )

        active_indices = still_active

        if not active_indices:
            break

        # ==================================================
        # CREATE ROOTS
        # ==================================================

        roots = [
            Node(
                boards[index]
            )
            for index in active_indices
        ]

        # ==================================================
        # BATCH ROOT EXPANSION
        # ==================================================
        #
        # All current game positions are evaluated
        # together on the GPU.
        #
        # ==================================================

        mcts._expand_roots_batched(
            roots
        )

        # ==================================================
        # DIRICHLET NOISE
        # ==================================================

        for root in roots:

            mcts.add_dirichlet_noise(

                root,

                alpha=dirichlet_alpha,

                epsilon=dirichlet_epsilon

            )

        # ==================================================
        # MULTI-GAME MCTS
        # ==================================================
        #
        # Each root receives exactly
        # num_simulations actual simulations.
        #
        # Neural-network evaluations from all games
        # are combined into GPU batches.
        #
        # ==================================================

        if num_simulations > 0:

            mcts.search_batched_multiple(

                roots,

                num_simulations=num_simulations,

                batch_size=batch_size

            )

        # ==================================================
        # PLAY ONE MOVE IN EACH GAME
        # ==================================================

        finished_this_round = []

        for root, index in zip(
            roots,
            active_indices
        ):

            board = boards[index]
            game = games[index]

            # --------------------------------------------------
            # Get MCTS policy target.
            # --------------------------------------------------

            policy = mcts.get_policy_target(
                root
            )

            # --------------------------------------------------
            # Store current position.
            # --------------------------------------------------

            game.add_position(
                board,
                policy
            )

            # --------------------------------------------------
            # Temperature.
            # --------------------------------------------------

            if (
                move_numbers[index]
                <= temperature_moves
            ):

                current_temperature = (
                    temperature
                )

            else:

                current_temperature = late_temperature

            # --------------------------------------------------
            # Select move.
            # --------------------------------------------------

            move = (
                mcts.select_action_with_temperature(

                    root,

                    temperature=current_temperature

                )
            )

            # --------------------------------------------------
            # Play move.
            # --------------------------------------------------

            board.push(
                move
            )

            move_numbers[index] += 1

            # --------------------------------------------------
            # Check whether game finished.
            # --------------------------------------------------

            if board.is_game_over(
                claim_draw=True
            ):

                results[index] = (
                    _finalize_game(
                        board,
                        game
                    )
                )

                finished_this_round.append(
                    index
                )

        # ==================================================
        # REMOVE FINISHED GAMES
        # ==================================================

        if finished_this_round:

            active_indices = [
                index
                for index in active_indices
                if index not in finished_this_round
            ]

    # ======================================================
    # PRINT STATISTICS
    # ======================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SELF-PLAY BATCH COMPLETE"
    )

    print(
        "=" * 60
    )

    total_samples = 0
    completed_games = 0
    incomplete_games = 0

    for index, result in enumerate(
        results
    ):

        if result is None:

            result = SelfPlayResult(

                training_data=[],
                result=None,
                termination="UNKNOWN",
                moves_played=len(
                    games[index].samples
                ),
                completed=False

            )

            results[index] = result

        if result.completed:

            completed_games += 1

        else:

            incomplete_games += 1

        total_samples += len(
            result.training_data
        )

        print(
            f"Game {index + 1}: "
            f"moves={result.moves_played} | "
            f"result={result.result} | "
            f"termination={result.termination} | "
            f"completed={result.completed}"
        )

    print(
        "-" * 60
    )

    print(
        "Games:",
        num_games
    )

    print(
        "Completed:",
        completed_games
    )

    print(
        "Incomplete:",
        incomplete_games
    )

    print(
        "Training samples:",
        total_samples
    )

    print(
        "=" * 60
    )

    return results


# ==========================================================
# PLAY ONE SELF-PLAY GAME
# ==========================================================
#
# Compatibility wrapper.
#
# Existing code that calls:
#
#     play_game(...)
#
# will continue to work.
#
# ==========================================================

def play_game(

    model,

    num_simulations=100,

    max_moves=200,

    temperature=1.0,

    temperature_moves=20,

    late_temperature=0.0,

    dirichlet_alpha=0.3,

    dirichlet_epsilon=0.25,

    batch_size=128

):

    results = play_games(

        model=model,

        num_games=1,

        num_simulations=num_simulations,

        max_moves=max_moves,

        temperature=temperature,

        temperature_moves=temperature_moves,

        late_temperature=late_temperature,

        dirichlet_alpha=dirichlet_alpha,

        dirichlet_epsilon=dirichlet_epsilon,

        batch_size=batch_size

    )

    return results[0]
