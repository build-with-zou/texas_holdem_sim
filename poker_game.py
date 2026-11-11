"""德州扑克下注阶段的基础状态和规则。

该模块只负责记录状态、判断动作是否合法并应用动作；玩家如何选择动作
留给后续策略模块实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from personalities import Personality


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class PlayerState:
    """单个玩家在一局中的筹码和行动状态。"""

    player_id: int
    stack: int | None
    folded: bool = False
    all_in: bool = False
    street_bet: int = 0
    committed: int = 0
    hole_cards: tuple[int, int] | None = None
    personality: Personality = Personality.TIGHT_PASSIVE

    @property
    def active(self) -> bool:
        return not self.folded and not self.all_in


@dataclass
class GameState:
    """一局牌在下注阶段的完整状态。金额均使用整数筹码。"""

    players: list[PlayerState]
    small_blind: int = 1
    big_blind: int = 2
    button: int = 0
    street: Street = Street.PREFLOP
    pot: int = 0
    current_bet: int = 0
    min_raise: int = 2
    community_cards: list[int] = field(default_factory=list)
    action_history: list[tuple[int, Action, int]] = field(default_factory=list)
    to_act: int | None = None
    max_raises: int | None = None
    raise_count: int = field(default=0, init=False)
    _pending: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if [p.player_id for p in self.players] != list(range(len(self.players))):
            raise ValueError("玩家编号必须按座位连续排列。")
        if any(p.stack is not None and (not isinstance(p.stack, int) or p.stack <= 0) for p in self.players):
            raise ValueError("筹码必须为正整数，或用 None 表示无上限。")
        if self.max_raises is not None and (not isinstance(self.max_raises, int) or self.max_raises < 0):
            raise ValueError("加注次数上限必须为非负整数。")
        if len(self.players) < 2:
            raise ValueError("至少需要两名玩家。")
        if self.small_blind <= 0 or self.big_blind < self.small_blind:
            raise ValueError("盲注必须为正数，且大盲注不小于小盲注。")
        if not 0 <= self.button < len(self.players):
            raise ValueError("button 必须对应一个玩家。")
        self.post_blinds()

    @property
    def active_players(self) -> list[PlayerState]:
        return [player for player in self.players if not player.folded]

    @property
    def contenders(self) -> list[PlayerState]:
        return [player for player in self.players if player.active]

    def _seat_after(self, seat: int) -> int:
        return (seat + 1) % len(self.players)

    def _next_active(self, seat: int, include_all_in: bool = False) -> int | None:
        for offset in range(1, len(self.players) + 1):
            candidate = self.players[(seat + offset) % len(self.players)]
            if not candidate.folded and (include_all_in or not candidate.all_in):
                return candidate.player_id
        return None

    def _commit(self, player: PlayerState, amount: int) -> int:
        amount = max(0, min(amount, player.stack)) if player.stack is not None else amount
        if player.stack is not None:
            player.stack -= amount
        player.street_bet += amount
        player.committed += amount
        self.pot += amount
        if player.stack == 0:
            player.all_in = True
        return amount

    def post_blinds(self) -> None:
        """发牌前收取盲注，并设置翻牌前第一个行动者。"""
        if len(self.players) == 2:
            sb = self.players[self.button]
            bb = self.players[(self.button + 1) % len(self.players)]
        else:
            sb = self.players[(self.button + 1) % len(self.players)]
            bb = self.players[(self.button + 2) % len(self.players)]
        self._commit(sb, self.small_blind)
        self._commit(bb, self.big_blind)
        self.current_bet = max(sb.street_bet, bb.street_bet)
        self.min_raise = self.big_blind
        self._pending = {p.player_id for p in self.players if p.active}
        self._set_next_actor(bb.player_id)

    def _set_next_actor(self, after: int) -> None:
        self._pending.intersection_update(p.player_id for p in self.players if p.active)
        if len(self.active_players) <= 1:
            self._pending.clear()
        if len(self.contenders) == 1:
            only = self.contenders[0]
            if only.street_bet >= self.current_bet:
                self._pending.clear()
        self.to_act = next(((after + offset) % len(self.players)
                            for offset in range(1, len(self.players) + 1)
                            if (after + offset) % len(self.players) in self._pending), None)

    @property
    def raises_allowed(self) -> bool:
        return self.max_raises is None or self.raise_count < self.max_raises

    def legal_actions(self, player_id: int | None = None) -> dict[Action, int]:
        """返回动作及其需要投入的筹码；数值为 0 的动作不代表投入筹码。"""
        player_id = self.to_act if player_id is None else player_id
        if player_id is None or player_id != self.to_act:
            return {}
        player = self.players[player_id]
        if not player.active:
            return {}
        to_call = self.current_bet - player.street_bet
        actions: dict[Action, int] = {Action.FOLD: 0}
        if to_call == 0:
            actions[Action.CHECK] = 0
        elif player.stack is None or player.stack > 0:
            actions[Action.CALL] = to_call if player.stack is None else min(to_call, player.stack)
        can_raise = self.raises_allowed and any(p.active and p.player_id != player_id for p in self.players)
        if can_raise and (player.stack is None or player.stack > to_call):
            if self.current_bet == 0:
                actions[Action.BET] = self.big_blind if player.stack is None else min(self.big_blind, player.stack)
            else:
                minimum = to_call + self.min_raise
                actions[Action.RAISE] = minimum if player.stack is None else min(minimum, player.stack)
        if player.stack is not None and player.stack > 0 and (can_raise or player.stack <= to_call):
            actions[Action.ALL_IN] = player.stack
        return actions

    def apply_action(self, action: Action | str, amount: int | None = None) -> int:
        """应用当前玩家动作，返回本次实际投入的筹码。"""
        if self.to_act is None:
            raise ValueError("当前没有可行动的玩家。")
        action = Action(action)
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"动作 {action.value} 在当前状态下不合法。")
        player = self.players[self.to_act]
        old_bet = self.current_bet
        required = legal[action]
        if action in (Action.BET, Action.RAISE):
            required = legal[action] if amount is None else amount
            if (not isinstance(required, int) or required < legal[action]
                    or (player.stack is not None and required > player.stack)):
                raise ValueError("下注须满足最小下注/加注额，且不能超过剩余筹码。")
        if amount is not None and action not in (Action.BET, Action.RAISE) and amount != required:
            raise ValueError("弃牌、过牌和跟注的投入由当前状态确定。")
        paid = self._commit(player, required)
        self._pending.discard(player.player_id)
        if action == Action.FOLD:
            player.folded = True
        elif action in (Action.BET, Action.RAISE, Action.ALL_IN):
            if player.street_bet > old_bet:
                raise_size = player.street_bet - old_bet
                if raise_size >= self.min_raise:
                    self.min_raise = raise_size
                self.current_bet = player.street_bet
                self.raise_count += 1
                self._pending.update(p.player_id for p in self.players
                                     if p.active and p.player_id != player.player_id
                                     and p.street_bet < self.current_bet)
        self.action_history.append((player.player_id, action, paid))
        self._set_next_actor(player.player_id)
        return paid

    def betting_round_complete(self) -> bool:
        return self.to_act is None

    def advance_street(self, community_cards: list[int] | None = None) -> None:
        """结束当前下注轮并进入下一街；摊牌前不自动发公共牌。"""
        if len(self.active_players) <= 1:
            raise ValueError("只剩一位玩家，应直接结算。")
        if not self.betting_round_complete():
            raise ValueError("当前下注轮尚未结束。")
        order = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
        if self.street == Street.RIVER:
            raise ValueError("河牌下注轮之后不能继续推进。")
        self.street = order[order.index(self.street) + 1]
        if community_cards:
            self.community_cards.extend(community_cards)
        for player in self.players:
            player.street_bet = 0
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.raise_count = 0
        self._pending = {p.player_id for p in self.players if p.active}
        self._set_next_actor(self.button)


def create_game(num_players: int, starting_stack: int | None = 100, button: int = 0,
                max_raises: int | None = None) -> GameState:
    """创建一局标准盲注、等筹码的游戏状态。"""
    if num_players < 2:
        raise ValueError("num_players 至少为 2。")
    if starting_stack is not None and starting_stack <= 0:
        raise ValueError("starting_stack 必须为正数。")
    return GameState(
        players=[PlayerState(player_id=i, stack=starting_stack) for i in range(num_players)],
        button=button,
        max_raises=max_raises,
    )
