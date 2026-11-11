"""计算指定人数下 169 类德州扑克起手牌的条件权益。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from holdem import RANK_LABELS, evaluate_seven


def card_id(rank: int, suit: int) -> int:
    return suit * 13 + rank - 2


def rank_label(rank: int) -> str:
    return RANK_LABELS[rank]


def starting_hand_classes() -> list[tuple[str, tuple[int, int], str]]:
    """返回 13 对子 + 78 同花 + 78 非同花，共 169 类起手牌。"""
    hands: list[tuple[str, tuple[int, int], str]] = []
    for high in range(14, 1, -1):
        for low in range(high, 1, -1):
            if high == low:
                label, kind = f"{rank_label(high)}{rank_label(low)}", "pair"
                cards = (card_id(high, 0), card_id(low, 1))
                hands.append((label, cards, kind))
            else:
                prefix = f"{rank_label(high)}{rank_label(low)}"
                hands.append((prefix + "s", (card_id(high, 0), card_id(low, 0)), "suited"))
                hands.append((prefix + "o", (card_id(high, 0), card_id(low, 1)), "offsuit"))
    return hands


def simulate_hand_equity(
    hero_cards: tuple[int, int], num_players: int, trials: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    deck = np.arange(52)
    remaining = deck[~np.isin(deck, hero_cards)]
    equity = outright_wins = ties = 0.0
    for _ in range(trials):
        dealt = rng.choice(remaining, size=2 * (num_players - 1) + 5, replace=False)
        opponents = dealt[: 2 * (num_players - 1)].reshape(num_players - 1, 2)
        board = dealt[2 * (num_players - 1) :]
        hero_score = evaluate_seven([*hero_cards, *board])
        opponent_scores = [evaluate_seven([*cards, *board]) for cards in opponents]
        best = max(hero_score, *opponent_scores)
        winners = sum(score == best for score in [hero_score, *opponent_scores])
        if hero_score == best:
            equity += 1 / winners
            if winners == 1:
                outright_wins += 1
            else:
                ties += 1
    return equity / trials, outright_wins / trials, ties / trials


def classify_strength(equity: float, num_players: int) -> str:
    ratio = equity / (1 / num_players)
    if ratio >= 1.5:
        return "strong"
    if ratio >= 0.75:
        return "medium"
    return "weak"


def run_experiment(num_players: int, trials: int, seed: int) -> pd.DataFrame:
    if num_players < 2 or num_players > 23:
        raise ValueError("玩家人数必须在 2 到 23 之间。")
    if trials <= 0:
        raise ValueError("trials 必须为正数。")
    rng = np.random.default_rng(seed)
    rows = []
    for label, cards, kind in starting_hand_classes():
        equity, win_probability, tie_probability = simulate_hand_equity(
            cards, num_players, trials, rng
        )
        ratio = equity * num_players
        rows.append(
            {
                "hand": label,
                "kind": kind,
                "equity": equity,
                "outright_win_probability": win_probability,
                "tie_probability": tie_probability,
                "relative_to_average": ratio,
                "strength": classify_strength(equity, num_players),
                "players": num_players,
                "trials": trials,
            }
        )
    return pd.DataFrame(rows).sort_values("equity", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="起手牌条件权益模拟")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--output", type=Path, default=Path("outputs/preflop_strength_6_players.csv"))
    args = parser.parse_args()
    result = run_experiment(args.players, args.trials, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(20).to_string(index=False))
    print(f"\n共生成 {len(result)} 类起手牌，结果写入：{args.output}")


if __name__ == "__main__":
    main()
