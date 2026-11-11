"""基于随机对手起手牌权益的翻牌前性格策略。"""
from __future__ import annotations

import csv
import math
from numbers import Integral
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


# 强、中、弱牌对价格变化的敏感程度；均为人为设定，可用于敏感性分析。
PRICE_SENSITIVITY = np.array((2., 6., 10.))
SIZE_FRACTIONS = np.array((.4, .8, 1.6))
PASSIVE_SIZES = ((.60, .30, .10), (.75, .20, .05), (.85, .13, .02))
AGGRESSIVE_SIZES = ((.20, .40, .40), (.40, .40, .20), (.55, .30, .15))


@dataclass(frozen=True)
class BettingContext:
    """行动前公开状态；金额为整数筹码，pot 包含此前所有投入。"""
    pot: int = 3
    current_bet: int = 2
    street_bet: int = 0
    min_raise: int = 2
    big_blind: int = 2

    def __post_init__(self):
        values = (self.pot, self.current_bet, self.street_bet, self.min_raise, self.big_blind)
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in values):
            raise ValueError('下注状态必须使用整数筹码。')
        if (self.pot < 0 or self.street_bet < 0 or self.current_bet < self.street_bet
                or self.min_raise <= 0 or self.big_blind <= 0
                or self.pot < self.current_bet):
            raise ValueError('底池、已下注额及最小加注额不合法。')

    @property
    def to_call(self) -> int:
        return self.current_bet - self.street_bet

    @property
    def call_price(self) -> float:
        return self.to_call / (self.pot + self.to_call) if self.to_call else 0.

    @property
    def raise_targets(self) -> tuple[int, int, int]:
        # 加注增量相对于跟注后的底池，并遵守上次完整加注的最小增量。
        base = max(self.pot + self.to_call, self.big_blind)
        minimum = self.min_raise if self.current_bet else self.big_blind
        increments = [max(minimum, math.ceil(f * base)) for f in SIZE_FRACTIONS]
        return tuple(self.current_bet + increment for increment in increments)

    @property
    def observations(self) -> tuple[str, ...]:
        passive = f'call_{self.to_call}' if self.to_call else 'check'
        aggressive = 'raise_to' if self.current_bet else 'bet_to'
        return tuple(dict.fromkeys(('fold', passive, *(f'{aggressive}_{n}' for n in self.raise_targets))))


def decision_likelihoods(personality: Personality, context: BettingContext = BettingContext()):
    """统一的动作+金额模型，返回观察名称和3×观察数的条件概率。"""
    personality = Personality(personality)
    params = POLICIES[personality]
    tight = personality in (Personality.TIGHT_PASSIVE, Personality.TIGHT_AGGRESSIVE)
    sensitive = PRICE_SENSITIVITY * (1.2 if tight else .7)
    if context.to_call:
        continuation = np.minimum(1., np.array(params.continue_rates)
                                  * np.exp(-sensitive * (context.call_price - .4)))
    else:
        continuation = np.ones(3)
    aggressive = np.array(params.aggressive_rates)
    passive_type = personality in (Personality.TIGHT_PASSIVE, Personality.LOOSE_PASSIVE)
    sizes = np.array(PASSIVE_SIZES if passive_type else AGGRESSIVE_SIZES)
    base = max(context.pot + context.to_call, context.big_blind)
    increments = np.array(context.raise_targets) - context.current_bet
    # 高于各档参考成本时折扣；放弃昂贵加注的概率归入跟注/过牌。
    acceptance = np.exp(-sensitive[:, None] * np.maximum(0., increments / base - SIZE_FRACTIONS))
    raises = continuation[:, None] * aggressive[:, None] * sizes * acceptance
    raw = np.column_stack((1 - continuation, continuation - raises.sum(axis=1), raises))
    passive = f'call_{context.to_call}' if context.to_call else 'check'
    prefix = 'raise_to' if context.current_bet else 'bet_to'
    raw_names = ('fold', passive, *(f'{prefix}_{n}' for n in context.raise_targets))
    names = context.observations
    # 最小加注限制可能使多个金额档合并；同一可见金额只能算作一个观察。
    model = np.column_stack([raw[:, [i for i, x in enumerate(raw_names) if x == name]].sum(axis=1)
                             for name in names])
    return names, model


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
    """返回当前行动者的分布；只支持翻牌前，使用统一的价格敏感模型。"""
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
    context = BettingContext(game.pot, game.current_bet, player.street_bet, game.min_raise, game.big_blind)
    names, model = decision_likelihoods(player.personality, context)
    free = Action.CHECK in legal
    passive = Action.CHECK if free else Action.CALL
    aggressive = Action.BET if Action.BET in legal else Action.RAISE
    probabilities = {action: 0.0 for action in legal}
    probabilities[Action.FOLD] = model[index, 0]
    probabilities[passive] = model[index, 1]
    for name, probability in zip(names[2:], model[index, 2:]):
        amount = int(name.rsplit('_', 1)[1]) - player.street_bet
        if aggressive in legal and legal[aggressive] <= amount and (player.stack is None or amount <= player.stack):
            probabilities[aggressive] += probability
        else:
            probabilities[passive] += probability
    return probabilities


def choose_action(game: GameState, table: PreflopTable, rng: np.random.Generator) -> Action:
    """兼容接口，仅返回动作；需要随机金额时使用 choose_decision。"""
    probabilities = action_probabilities(game, table)
    actions = list(probabilities)
    return actions[int(rng.choice(len(actions), p=list(probabilities.values())))]


def choose_decision(game: GameState, table: PreflopTable, rng: np.random.Generator) -> tuple[Action, int]:
    """选择动作及本次额外投入；供 apply_action(action, amount) 使用。"""
    from poker_game import Action
    probabilities = action_probabilities(game, table)
    actions = list(probabilities)
    action = actions[int(rng.choice(len(actions), p=list(probabilities.values())))]
    if action not in (Action.BET, Action.RAISE):
        return action, game.legal_actions()[action]
    player = game.players[game.to_act]
    context = BettingContext(game.pot, game.current_bet, player.street_bet, game.min_raise, game.big_blind)
    names, model = decision_likelihoods(player.personality, context)
    index = ('strong', 'medium', 'weak').index(table.lookup(player.hole_cards, len(game.players)).strength)
    candidates = [(int(name.rsplit('_', 1)[1]) - player.street_bet, p)
                  for name, p in zip(names[2:], model[index, 2:])
                  if game.legal_actions()[action] <= int(name.rsplit('_', 1)[1]) - player.street_bet
                  and (player.stack is None or int(name.rsplit('_', 1)[1]) - player.street_bet <= player.stack)]
    weights = np.array([p for _, p in candidates])
    chosen = int(rng.choice(len(candidates), p=weights / weights.sum()))
    return action, candidates[chosen][0]
