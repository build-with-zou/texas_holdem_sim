from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from holdem import HAND_CATEGORY_NAMES, HAND_CATEGORY_NAMES_ZH, deal_showdown, run_evaluator_self_test


DEFAULT_PLAYER_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 10]


def simulate_player_count(num_players: int, trials: int, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    hero_equity = 0.0
    hero_outright_wins = 0
    hero_ties = 0
    tied_games = 0
    total_winners = 0
    winning_category_counts = {category: 0 for category in HAND_CATEGORY_NAMES}

    for _ in range(trials):
        result = deal_showdown(num_players, rng)
        winners = result["winners"]
        winning_category = result["winning_score"][0]
        winner_count = len(winners)

        winning_category_counts[winning_category] += 1
        total_winners += winner_count

        if winner_count > 1:
            tied_games += 1

        if 0 in winners:
            hero_equity += 1 / winner_count
            if winner_count == 1:
                hero_outright_wins += 1
            else:
                hero_ties += 1

    summary = {
        "player_count": num_players,
        "trials": trials,
        "hero_equity_probability": hero_equity / trials,
        "hero_outright_win_probability": hero_outright_wins / trials,
        "hero_tie_probability": hero_ties / trials,
        "table_tie_rate": tied_games / trials,
        "average_winner_count": total_winners / trials,
        "theoretical_equal_equity": 1 / num_players,
    }

    distribution_rows = []
    for category, count in winning_category_counts.items():
        distribution_rows.append(
            {
                "player_count": num_players,
                "hand_category": category,
                "hand_category_name": HAND_CATEGORY_NAMES[category],
                "hand_category_name_zh": HAND_CATEGORY_NAMES_ZH[category],
                "count": count,
                "probability": count / trials,
            }
        )

    return summary, pd.DataFrame(distribution_rows)


def run_simulation(player_counts: list[int], trials: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summary_rows = []
    distribution_frames = []

    for num_players in player_counts:
        summary, distribution = simulate_player_count(num_players, trials, rng)
        summary_rows.append(summary)
        distribution_frames.append(distribution)
        print(
            f"{num_players:>2} players | "
            f"hero equity={summary['hero_equity_probability']:.4f}, "
            f"outright={summary['hero_outright_win_probability']:.4f}, "
            f"tie rate={summary['table_tie_rate']:.4f}"
        )

    return pd.DataFrame(summary_rows), pd.concat(distribution_frames, ignore_index=True)


def plot_win_probabilities(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5.2))
    plt.plot(
        summary["player_count"],
        summary["hero_equity_probability"],
        marker="o",
        label="Hero equity with split pots",
    )
    plt.plot(
        summary["player_count"],
        summary["hero_outright_win_probability"],
        marker="s",
        label="Hero outright win probability",
    )
    plt.plot(
        summary["player_count"],
        summary["theoretical_equal_equity"],
        linestyle="--",
        color="gray",
        label="1 / number of players",
    )
    plt.title("Texas Hold'em Pure Showdown Model")
    plt.xlabel("Number of players")
    plt.ylabel("Probability")
    plt.xticks(summary["player_count"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "win_probability_by_player_count.png", dpi=180)
    plt.close()


def plot_tie_rate(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5.2))
    plt.plot(summary["player_count"], summary["table_tie_rate"], marker="o", color="#8a3ffc")
    plt.title("Tie Rate Increases With More Players")
    plt.xlabel("Number of players")
    plt.ylabel("Table tie rate")
    plt.xticks(summary["player_count"])
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "tie_rate_by_player_count.png", dpi=180)
    plt.close()


def plot_winning_hand_distribution(distribution: pd.DataFrame, output_dir: Path, selected_player_count: int) -> None:
    subset = distribution[distribution["player_count"] == selected_player_count].sort_values("hand_category")
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(subset)))

    plt.figure(figsize=(10.5, 5.5))
    plt.bar(subset["hand_category_name"], subset["probability"], color=colors)
    plt.title(f"Winning Hand Distribution ({selected_player_count} Players)")
    plt.xlabel("Winning hand category")
    plt.ylabel("Probability")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"winning_hand_distribution_{selected_player_count}_players.png", dpi=180)
    plt.close()


def write_outputs(summary: pd.DataFrame, distribution: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_chart in output_dir.glob("winning_hand_distribution_*_players.png"):
        stale_chart.unlink()

    summary.to_csv(output_dir / "summary_by_player_count.csv", index=False)
    distribution.to_csv(output_dir / "winning_hand_distribution.csv", index=False)

    plot_win_probabilities(summary, output_dir)
    plot_tie_rate(summary, output_dir)
    plot_winning_hand_distribution(
        distribution,
        output_dir,
        selected_player_count=int(summary["player_count"].max()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-probability Texas Hold'em showdown simulator."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5_000,
        help="Number of simulated games for each player count.",
    )
    parser.add_argument(
        "--players",
        type=int,
        nargs="+",
        default=DEFAULT_PLAYER_COUNTS,
        help="Player counts to simulate, e.g. --players 2 6 9.",
    )
    parser.add_argument("--seed", type=int, default=261, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV files and charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluator_self_test()

    summary, distribution = run_simulation(
        player_counts=args.players,
        trials=args.trials,
        seed=args.seed,
    )
    write_outputs(summary, distribution, args.output_dir)
    print(f"\nDone. Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
