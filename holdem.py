from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

import numpy as np


RANK_LABELS = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}

SUIT_LABELS = ["c", "d", "h", "s"]

HAND_CATEGORY_NAMES = {
    0: "High Card",
    1: "One Pair",
    2: "Two Pair",
    3: "Three of a Kind",
    4: "Straight",
    5: "Flush",
    6: "Full House",
    7: "Four of a Kind",
    8: "Straight Flush",
}

HAND_CATEGORY_NAMES_ZH = {
    0: "高牌",
    1: "一对",
    2: "两对",
    3: "三条",
    4: "顺子",
    5: "同花",
    6: "葫芦",
    7: "四条",
    8: "同花顺",
}

Score = tuple[int, tuple[int, ...]]


def card_rank(card: int) -> int:
    """Return rank from 2 to 14, where 14 means Ace."""
    return card % 13 + 2


def card_suit(card: int) -> int:
    return card // 13


def card_to_str(card: int) -> str:
    return f"{RANK_LABELS[card_rank(card)]}{SUIT_LABELS[card_suit(card)]}"


def cards_to_str(cards: Iterable[int]) -> str:
    return " ".join(card_to_str(card) for card in cards)


def _straight_high(ranks: Iterable[int]) -> int | None:
    rank_set = set(ranks)
    if 14 in rank_set:
        rank_set.add(1)

    for high in range(14, 4, -1):
        if all((high - offset) in rank_set for offset in range(5)):
            return high
    return None


def evaluate_five(cards: Iterable[int]) -> Score:
    cards = list(cards)
    ranks = [card_rank(card) for card in cards]
    suits = [card_suit(card) for card in cards]
    counts = Counter(ranks)
    count_groups = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if is_flush and straight_high is not None:
        return 8, (straight_high,)

    if count_groups[0][1] == 4:
        quad_rank = count_groups[0][0]
        kicker = max(rank for rank in ranks if rank != quad_rank)
        return 7, (quad_rank, kicker)

    if count_groups[0][1] == 3 and count_groups[1][1] == 2:
        return 6, (count_groups[0][0], count_groups[1][0])

    if is_flush:
        return 5, tuple(sorted(ranks, reverse=True))

    if straight_high is not None:
        return 4, (straight_high,)

    if count_groups[0][1] == 3:
        trip_rank = count_groups[0][0]
        kickers = sorted((rank for rank in ranks if rank != trip_rank), reverse=True)
        return 3, (trip_rank, *kickers)

    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = max(rank for rank in ranks if rank not in pairs)
        return 2, (*pairs, kicker)

    if len(pairs) == 1:
        pair_rank = pairs[0]
        kickers = sorted((rank for rank in ranks if rank != pair_rank), reverse=True)
        return 1, (pair_rank, *kickers)

    return 0, tuple(sorted(ranks, reverse=True))


def evaluate_seven(cards: Iterable[int]) -> Score:
    cards = list(cards)
    if len(cards) != 7:
        raise ValueError(f"Texas Hold'em evaluator expects 7 cards, got {len(cards)}.")
    return max(evaluate_five(combo) for combo in combinations(cards, 5))


def deal_showdown(num_players: int, rng: np.random.Generator) -> dict:
    if not 2 <= num_players <= 23:
        raise ValueError("num_players must be between 2 and 23.")

    deck = rng.permutation(52)
    hole_cards = deck[: 2 * num_players].reshape(num_players, 2)
    community_cards = deck[2 * num_players : 2 * num_players + 5]
    scores = [
        evaluate_seven([*hole_cards[player_idx], *community_cards])
        for player_idx in range(num_players)
    ]
    best_score = max(scores)
    winners = [idx for idx, score in enumerate(scores) if score == best_score]

    return {
        "hole_cards": hole_cards,
        "community_cards": community_cards,
        "scores": scores,
        "winners": winners,
        "winning_score": best_score,
    }


def run_evaluator_self_test() -> None:
    # Card ids are suit * 13 + (rank - 2). These checks guard the important edge cases.
    royal_flush = [8, 9, 10, 11, 12]  # Tc Jc Qc Kc Ac
    four_aces = [12, 25, 38, 51, 7]  # Ac Ad Ah As 9c
    wheel = [12, 0, 1, 2, 3]  # Ac 2c 3c 4c 5c
    full_house = [10, 23, 36, 7, 20]  # Qc Qd Qh 9c 9d

    assert evaluate_five(royal_flush)[0] == 8
    assert evaluate_five(four_aces)[0] == 7
    assert evaluate_five(wheel) == (8, (5,))
    assert evaluate_five(full_house) == (6, (12, 9))
    assert evaluate_five(royal_flush) > evaluate_five(four_aces)
