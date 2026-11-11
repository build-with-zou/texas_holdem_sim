"""基于随机对手起手牌权益的翻牌前性格策略。"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from holdem import RANK_LABELS, card_rank, card_suit

if TYPE_CHECKING:
    from poker_game import Action, GameState


class Personality(str, Enum):
    TIGHT_PASSIVE = 'tight_passive'
    TIGHT_AGGRESSIVE = 'tight_aggressive'
    LOOSE_PASSIVE = 'loose_passive'
    LOOSE_AGGRESSIVE = 'loose_aggressive'


@dataclass(frozen=True)
class PolicyParameters:
    # 顺序均为 strong、medium、weak。
    continue_rates: tuple[float, float, float]
    aggressive_rates: tuple[float, float, float]


POLICIES = {
    Personality.TIGHT_PASSIVE: PolicyParameters((0.95, 0.20, 0.05), (0.10, 0.05, 0.01)),
    Personality.TIGHT_AGGRESSIVE: PolicyParameters((0.95, 0.20, 0.05), (0.80, 0.50, 0.20)),
    Personality.LOOSE_PASSIVE: PolicyParameters((0.99, 0.80, 0.40), (0.10, 0.05, 0.01)),
    Personality.LOOSE_AGGRESSIVE: PolicyParameters((0.99, 0.80, 0.40), (0.80, 0.50, 0.20)),
}


def hand_class(cards: tuple[int, int]) -> str:
    """将本项目的两张牌 ID 转为 AA、AKs、AKo 等类别。"""
    if len(cards) != 2 or any(not 0 <= c < 52 for c in cards) or cards[0] == cards[1]:
        raise ValueError('底牌必须是两张不同的有效牌（ID 0–51）。')
    high, low = sorted((card_rank(c) for c in cards), reverse=True)
    label = RANK_LABELS[high] + RANK_LABELS[low]
    if high != low:
        label += 's' if card_suit(cards[0]) == card_suit(cards[1]) else 'o'
    return label


@dataclass(frozen=True)
class HandStrength:
    hand: str
    players: int
    equity: float
    strength: str


class PreflopTable:
    """按本局初始人数查表；弃牌后不切换到随机对手的另一张表。"""
    def __init__(self, path: Path | str | None = None):
        path = Path(path) if path is not None else Path(__file__).resolve().parent / 'outputs/preflop_by_players/preflop_strength_all_players.csv'
        self.entries: dict[tuple[int, str], HandStrength] = {}
        with path.open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                n, equity = int(row['players']), float(row['equity'])
                if not math.isfinite(equity) or not 0 <= equity <= 1:
                    raise ValueError('权益必须在 0 到 1 之间。')
                strength = 'strong' if equity * n >= 1.5 else 'medium' if equity * n >= 0.75 else 'weak'
                key = (n, row['hand'])
                if key in self.entries:
                    raise ValueError(f'重复记录：{key}')
                self.entries[key] = HandStrength(row['hand'], n, equity, strength)

    def lookup(self, cards: tuple[int, int], players: int) -> HandStrength:
        key = (players, hand_class(cards))
        try:
            return self.entries[key]
        except KeyError as exc:
            raise ValueError(f'强度表缺少 {key}，请先生成对应人数的实验数据。') from exc


def action_probabilities(game: GameState, table: PreflopTable) -> dict[Action, float]:
    """返回当前行动者的分布；只支持翻牌前，只选现有框架允许的最小加注。"""
    from poker_game import Action, Street

    if game.street != Street.PREFLOP:
        raise ValueError('起手牌性格策略仅用于翻牌前。')
    legal = game.legal_actions()
    if not legal:
        raise ValueError('当前没有可行动的玩家。')
    player = game.players[game.to_act]
    if player.hole_cards is None:
        raise ValueError('请先设置当前玩家的 hole_cards。')
    value = table.lookup(player.hole_cards, len(game.players))
    index = ('strong', 'medium', 'weak').index(value.strength)
    params = POLICIES[Personality(player.personality)]
    free = Action.CHECK in legal
    continuation = 1.0 if free else params.continue_rates[index]
    aggression = params.aggressive_rates[index]
    passive = Action.CHECK if free else Action.CALL
    aggressive = Action.BET if Action.BET in legal else Action.RAISE
    probabilities = {action: 0.0 for action in legal}
    probabilities[Action.FOLD] = 1 - continuation
    if aggressive in legal:
        probabilities[aggressive] = continuation * aggression
        probabilities[passive] = continuation * (1 - aggression)
    else:
        probabilities[passive] = continuation
    # 不单独随机选择 ALL_IN；短筹码跟注或最小加注可能自然耗尽筹码。
    return probabilities


def choose_action(game: GameState, table: PreflopTable, rng: np.random.Generator) -> Action:
    probabilities = action_probabilities(game, table)
    actions = list(probabilities)
    return actions[int(rng.choice(len(actions), p=list(probabilities.values())))]
