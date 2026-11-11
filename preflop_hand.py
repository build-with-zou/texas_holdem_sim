"""有限次翻牌前下注后直接摊牌的最小可结算牌局，无筹码上限、无全押。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from holdem import cards_to_str, evaluate_seven
from personalities import BettingContext, Personality, PreflopTable, decision_likelihoods
from bayes_preflop import STRENGTHS, likelihoods, prior_from_cards, update_posterior
from poker_game import Action, Street, create_game


@dataclass(frozen=True)
class PublicPlayer:
    player_id: int
    position: str
    personality: Personality
    folded: bool
    street_bet: int
    committed: int


@dataclass(frozen=True)
class PublicAction:
    player_id: int
    action: Action
    paid: int
    target: int
    context_before: BettingContext
    pot_after: int


@dataclass(frozen=True)
class DecisionView:
    """策略唯一输入：本人的底牌、已知性格和公开下注信息。"""
    player_id: int
    hole_cards: tuple[int, int]
    players: tuple[PublicPlayer, ...]
    button: int
    community_cards: tuple[int, ...]
    street: Street
    context: BettingContext
    legal_actions: tuple[tuple[Action, int], ...]
    raises_remaining: int
    history: tuple[PublicAction, ...]


def positions(num_players: int, button: int) -> tuple[str, ...]:
    labels = [''] * num_players
    if num_players == 2:
        labels[button] = 'BTN/SB'
        labels[(button + 1) % num_players] = 'BB'
    else:
        labels[button] = 'BTN'
        labels[(button + 1) % num_players] = 'SB'
        labels[(button + 2) % num_players] = 'BB'
        for offset in range(3, num_players):
            labels[(button + offset) % num_players] = ('UTG/CO' if num_players == 4 else
                                                     'UTG' if offset == 3 else
                                                     'CO' if offset == num_players - 1 else
                                                     'HJ' if offset == num_players - 2 else f'UTG+{offset - 3}')
    return tuple(labels)


class PreflopHand:
    def __init__(self, personalities: list[Personality], rng: np.random.Generator,
                 button: int = 0, max_raises: int = 3, deck: list[int] | None = None):
        if not 2 <= len(personalities) <= 10:
            raise ValueError('最小牌局支持2–10人。')
        if not isinstance(max_raises, int) or isinstance(max_raises, bool) or max_raises < 0:
            raise ValueError('必须设置非负整数的加注次数上限。')
        self._game = create_game(len(personalities), starting_stack=None, button=button, max_raises=max_raises)
        deck = list(map(int, rng.permutation(52))) if deck is None else list(deck)
        if (len(deck) != 52 or any(not isinstance(c, int) or isinstance(c, bool) for c in deck)
                or set(deck) != set(range(52))):
            raise ValueError('牌堆必须包含0–51且无重复。')
        for i, (player, personality) in enumerate(zip(self._game.players, personalities)):
            player.personality = Personality(personality)
            player.hole_cards = tuple(deck[2 * i:2 * i + 2])
        self._board = tuple(deck[2 * len(personalities):2 * len(personalities) + 5])
        self._positions = positions(len(personalities), button)
        self._history: list[PublicAction] = []
        self._settled = False

    @property
    def complete(self) -> bool:
        return self._game.betting_round_complete()

    def view(self) -> DecisionView:
        if self.complete or self._settled:
            raise ValueError('当前没有待决策玩家。')
        g = self._game
        player = g.players[g.to_act]
        return DecisionView(player.player_id, player.hole_cards,
                            tuple(PublicPlayer(p.player_id, self._positions[p.player_id], p.personality,
                                               p.folded, p.street_bet, p.committed) for p in g.players),
                            g.button, tuple(g.community_cards), g.street,
                            BettingContext(g.pot, g.current_bet, player.street_bet, g.min_raise, g.big_blind),
                            tuple(g.legal_actions().items()), g.max_raises - g.raise_count,
                            tuple(self._history))

    def apply(self, action: Action | str, amount: int | None = None) -> None:
        view = self.view()
        action = Action(action)
        paid = self._game.apply_action(action, amount)
        self._history.append(PublicAction(view.player_id, action, paid,
                                         self._game.players[view.player_id].street_bet,
                                         view.context, self._game.pot))

    def settle(self) -> dict:
        if self._settled:
            raise ValueError('本局已经结算，不能重复发放底池。')
        if not self.complete:
            raise ValueError('下注轮未结束，不能结算。')
        g = self._game
        # 最小牌局不允许全押，存活玩家必须已经跟齐；无边池。
        alive = [p for p in g.players if not p.folded]
        if len(alive) > 1 and any(p.committed != g.current_bet for p in alive):
            raise RuntimeError('存活玩家未跟齐。')
        gross = [p.committed for p in g.players]
        refunds = [0] * len(g.players)
        highest = max(gross)
        if gross.count(highest) == 1:
            owner = gross.index(highest)
            refunds[owner] = highest - sorted(gross)[-2]
        contested_pot = sum(gross) - sum(refunds)
        scores = {}
        if len(alive) == 1:
            winners = [alive[0].player_id]
            board = ()
            reason = 'all_others_folded'
        else:
            board = self._board
            scores = {p.player_id: evaluate_seven((*p.hole_cards, *board)) for p in alive}
            best = max(scores.values())
            winners = [i for i, score in scores.items() if score == best]
            reason = 'showdown'
        awards = [0] * len(g.players)
        share, odd_chips = divmod(contested_pot, len(winners))
        # 平分后的余数筹码，从按钮左侧开始按座位依次分配。
        ordered_winners = sorted(winners, key=lambda i: (i - g.button - 1) % len(g.players))
        for i in ordered_winners:
            awards[i] = share
        for i in ordered_winners[:odd_chips]:
            awards[i] += 1
        net = [awards[i] + refunds[i] - gross[i] for i in range(len(g.players))]
        if sum(net) != 0 or sum(awards) + sum(refunds) != sum(gross):
            raise RuntimeError('结算未满足筹码守恒。')
        self._settled = True
        return dict(reason=reason, button=g.button, small_blind=g.small_blind, big_blind=g.big_blind,
                    max_raises=g.max_raises, raises_used=g.raise_count,
                    community_cards=list(board), community_cards_text=cards_to_str(board),
                    gross_pot=sum(gross), uncalled_refund=sum(refunds), contested_pot=contested_pot,
                    winners=winners,
                    players=[dict(player_id=p.player_id, position=self._positions[p.player_id],
                                  personality=p.personality.value, hole_cards=list(p.hole_cards),
                                  hole_cards_text=cards_to_str(p.hole_cards), folded=p.folded,
                                  committed=gross[p.player_id], refund=refunds[p.player_id],
                                  award=awards[p.player_id], net_profit=net[p.player_id],
                                  score=scores.get(p.player_id)) for p in g.players],
                    actions=[asdict(event) for event in self._history])


def personality_distribution(view: DecisionView, table: PreflopTable) -> dict[tuple[Action, int], float]:
    """仅从可见信息生成概率，达到加注上限后的概率转入跟注/过牌。"""
    if view.street != Street.PREFLOP:
        raise ValueError('当前策略仅支持翻牌前。')
    strength = table.lookup(view.hole_cards, len(view.players)).strength
    index = ('strong', 'medium', 'weak').index(strength)
    personality = view.players[view.player_id].personality
    names, model = decision_likelihoods(personality, view.context)
    legal = dict(view.legal_actions)
    passive = Action.CALL if view.context.to_call else Action.CHECK
    aggressive = Action.RAISE if view.context.current_bet else Action.BET
    probabilities = {(Action.FOLD, 0): float(model[index, 0]),
                     (passive, legal[passive]): float(model[index, 1])}
    for name, p in zip(names[2:], model[index, 2:]):
        amount = int(name.rsplit('_', 1)[1]) - view.context.street_bet
        decision = (aggressive, amount) if aggressive in legal else (passive, legal[passive])
        probabilities[decision] = probabilities.get(decision, 0.) + float(p)
    return probabilities


def personality_decision(view: DecisionView, rng: np.random.Generator, table: PreflopTable):
    distribution = personality_distribution(view, table)
    choices = list(distribution)
    return choices[int(rng.choice(len(choices), p=list(distribution.values())))]


def opponent_strength_posteriors(view: DecisionView, table: PreflopTable | None = None) -> dict[int, np.ndarray]:
    """根据已观察到的公开行动，顺序更新每个对手的强/中/弱后验。"""
    prior = np.full(3, 1 / 3)
    if table is not None:
        prior, _, _, _ = prior_from_cards(table, view.hole_cards, len(view.players))
    posteriors = {p.player_id: prior.copy() for p in view.players
                  if p.player_id != view.player_id and not p.folded}
    for event in view.history:
        if event.player_id not in posteriors:
            continue
        names, model = decision_likelihoods(
            view.players[event.player_id].personality,
            event.context_before,
        )
        label = 'fold' if event.action == Action.FOLD else (
            ('check' if event.action == Action.CHECK else
             f'call_{event.paid}' if event.action == Action.CALL else
             f'raise_to_{event.target}'))
        if label not in names:
            continue
        posteriors[event.player_id] = update_posterior(
            posteriors[event.player_id], model, names.index(label)
        )
    return posteriors


def data_driven_decision(view: DecisionView, rng: np.random.Generator,
                         table: PreflopTable) -> tuple[Action, int]:
    """基于自己的条件权益和对手行动后验的简化决策策略。

    对手底牌仍不可见；性格、下注上下文和历史行动可见。策略使用行动后验
    提高对强牌对手的风险估计，再与当前价格比较。
    """
    legal = dict(view.legal_actions)
    passive = Action.CALL if view.context.to_call else Action.CHECK
    aggressive = Action.RAISE if view.context.current_bet else Action.BET
    own = table.lookup(view.hole_cards, len(view.players))
    own_equity = own.equity
    posteriors = opponent_strength_posteriors(view, table)
    # 强牌后验权重最高；没有观察到行动的对手按中性风险处理。
    danger = sum(float(p[0] + .5 * p[1]) for p in posteriors.values())
    danger /= max(1, len(posteriors))
    estimated_equity = own_equity * (1 - .35 * danger)
    price = view.context.call_price
    if passive in legal and view.context.to_call and estimated_equity < price:
        return Action.FOLD, legal[Action.FOLD]
    if aggressive in legal and estimated_equity >= max(.35, 1.5 * price):
        targets = [n for n in view.context.raise_targets
                   if legal[aggressive] <= n - view.context.street_bet]
        if targets:
            # 优势越大越倾向使用更大的公开金额，但仍随机化相邻档位。
            target = targets[-1] if estimated_equity >= .55 else targets[0]
            return aggressive, target - view.context.street_bet
    if passive in legal:
        return passive, legal[passive]
    return Action.FOLD, legal[Action.FOLD]


def random_decision(view: DecisionView, rng: np.random.Generator):
    """不看底牌：各可选动作等概率，加注时各个不同金额等概率。"""
    legal = dict(view.legal_actions)
    actions = [a for a in legal if a != Action.ALL_IN and not (a == Action.FOLD and Action.CHECK in legal)]
    action = actions[int(rng.integers(len(actions)))]
    if action in (Action.BET, Action.RAISE):
        targets = tuple(dict.fromkeys(view.context.raise_targets))
        return action, targets[int(rng.integers(len(targets)))] - view.context.street_bet
    return action, legal[action]


def play_hand(hand: PreflopHand, policies: list[Callable], action_rngs: list[np.random.Generator]) -> dict:
    if len(policies) != len(action_rngs) or len(policies) != len(hand._game.players):
        raise ValueError('每个座位都必须有独立策略及随机数流。')
    # 最多每个加注区间每人行动一次；作为异常保护，不能代替下注轮规则。
    limit = len(policies) * (hand._game.max_raises + 1)
    steps = 0
    while not hand.complete:
        if steps >= limit:
            raise RuntimeError('超过有限加注模型的最大行动次数。')
        view = hand.view()
        action, amount = policies[view.player_id](view, action_rngs[view.player_id])
        hand.apply(action, amount)
        steps += 1
    return hand.settle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players', type=int, choices=range(2, 11), default=6)
    parser.add_argument('--hands', type=int, default=1)
    parser.add_argument('--seed', type=int, default=261)
    parser.add_argument('--max-raises', type=int, default=3)
    parser.add_argument('--policy', choices=('random', 'personality', 'informed', 'mixed'), default='mixed')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/minimal_preflop'))
    args = parser.parse_args()
    if args.hands <= 0:
        parser.error('--hands 必须为正数。')
    table = PreflopTable()
    personalities = [list(Personality)[i % 4] for i in range(args.players)]
    def personality_policy(view, rng):
        return personality_decision(view, rng, table)
    def informed_policy(view, rng):
        return data_driven_decision(view, rng, table)
    policy_names = [
        ('random' if args.policy == 'random' else
         'informed' if args.policy == 'informed' or (args.policy == 'mixed' and i == 0)
         else 'personality') for i in range(args.players)
    ]
    policies = [
        (random_decision if args.policy == 'random' else
         informed_policy if args.policy == 'informed' or (args.policy == 'mixed' and i == 0)
         else personality_policy) for i in range(args.players)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_streams = np.random.SeedSequence(args.seed).spawn(args.hands)
    net = np.zeros(args.players, dtype=np.int64)
    endings = {'showdown': 0, 'all_others_folded': 0}
    with (args.output_dir / 'hands.jsonl').open('w', encoding='utf-8') as file:
        for number, stream in enumerate(seed_streams):
            streams = stream.spawn(args.players + 1)
            hand = PreflopHand(personalities, np.random.default_rng(streams[0]),
                               button=number % args.players, max_raises=args.max_raises)
            result = play_hand(hand, policies, [np.random.default_rng(s) for s in streams[1:]])
            result.update(hand_id=number, seed=args.seed, policy=args.policy)
            for player, policy_name in zip(result['players'], policy_names):
                player['policy'] = policy_name
            net += [p['net_profit'] for p in result['players']]
            endings[result['reason']] += 1
            file.write(json.dumps(result, ensure_ascii=False) + '\n')
            if args.hands == 1:
                print(json.dumps(result, ensure_ascii=False, indent=2))
    summary = dict(hands=args.hands, players=args.players, seed=args.seed, policy=args.policy,
                   max_raises=args.max_raises, endings=endings, net_profit=net.tolist(),
                   mean_profit_per_hand=(net / args.hands).tolist(), zero_sum=int(net.sum()) == 0)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
