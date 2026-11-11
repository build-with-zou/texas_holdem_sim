"""六人桌首次行动实验：已知性格，从动作与金额推断底牌等级。

这是独立的无筹码上限、单次观察实验，不调用完整牌局引擎。
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from personalities import (Personality, PreflopTable, hand_class, BettingContext,
                           decision_likelihoods)

STRENGTHS = ('strong', 'medium', 'weak')
OBSERVATIONS = BettingContext().observations


def likelihoods(personality: Personality, context: BettingContext = BettingContext()) -> np.ndarray:
    """返回 P(动作与金额 | 牌力, 性格, 行动前公开状态)。"""
    return decision_likelihoods(personality, context)[1]


def prior_from_cards(table: PreflopTable, hero_cards: tuple[int, int], players: int = 6):
    """等概率枚举排除观察者底牌后的实际两张牌组合，而非等概率枚举169类。"""
    hand_class(hero_cards)  # 校验底牌
    deck = [card for card in range(52) if card not in hero_cards]
    hands = list(combinations(deck, 2))
    labels = np.array([STRENGTHS.index(table.lookup(cards, players).strength) for cards in hands])
    counts = np.bincount(labels, minlength=3)
    return counts / len(hands), counts, hands, labels


def update_posterior(prior: np.ndarray, likelihood: np.ndarray, observation: int) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if prior.shape != (3,) or likelihood.ndim != 2 or likelihood.shape[0] != 3:
        raise ValueError('先验须有三项，似然须为3×观察数矩阵。')
    if (not np.isfinite(prior).all() or not np.isfinite(likelihood).all()
            or (prior < 0).any() or (likelihood < 0).any()
            or not np.isclose(prior.sum(), 1) or not np.allclose(likelihood.sum(axis=1), 1)):
        raise ValueError('先验和似然必须为有效概率分布。')
    if not 0 <= observation < likelihood.shape[1]:
        raise ValueError('无效观察编号。')
    weights = prior * likelihood[:, observation]
    if weights.sum() == 0:
        raise ValueError('该观察在模型下概率为零，无法更新。')
    return weights / weights.sum()


def run_experiment(trials: int = 100_000, seed: int = 261,
                   output_dir: Path = Path('outputs/bayes_6_players'),
                   context: BettingContext = BettingContext()) -> pd.DataFrame:
    if trials <= 0:
        raise ValueError('模拟次数必须为正数。')
    observations = context.observations
    table = PreflopTable()
    hero_cards = (12, 11)  # Ac Kc
    prior, counts, hands, labels = prior_from_cards(table, hero_cards)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'strength': STRENGTHS, 'combinations': counts, 'prior': prior}).to_csv(
        output_dir / 'prior.csv', index=False)
    rows = []
    streams = np.random.SeedSequence(seed).spawn(len(Personality))
    for personality, stream in zip(Personality, streams):
        rng = np.random.default_rng(stream)
        model = likelihoods(personality, context)
        # 从1225个真实底牌组合等概率抽样；推断接口不接收这些隐藏底牌。
        true_labels = labels[rng.integers(0, len(hands), size=trials)]
        uniforms = rng.random(trials)
        events = (uniforms[:, None] > np.cumsum(model[true_labels], axis=1)).sum(axis=1)
        for observation, name in enumerate(observations):
            if prior @ model[:, observation] == 0:
                continue
            posterior = update_posterior(prior, model, observation)
            selected = true_labels[events == observation]
            empirical = np.bincount(selected, minlength=3) / len(selected) if len(selected) else np.full(3, np.nan)
            for i, strength in enumerate(STRENGTHS):
                rows.append(dict(personality=personality.value, observation=name,
                                 strength=strength, prior=prior[i],
                                 likelihood=model[i, observation],
                                 observation_probability=float(prior @ model[:, observation]),
                                 posterior=posterior[i], empirical_posterior=empirical[i],
                                 observation_count=len(selected), trials=trials, seed=seed, pot=context.pot, to_call=context.to_call,
                                 current_bet=context.current_bet, street_bet=context.street_bet,
                                 min_raise=context.min_raise, big_blind=context.big_blind))
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / 'posterior_comparison.csv', index=False)
    plot_results(result, output_dir, observations)
    print('Prior (Ac Kc excluded):', dict(zip(STRENGTHS, prior)))
    strong = result[result.strength == 'strong']
    print(strong.pivot(index='personality', columns='observation', values='posterior').to_string())
    print('Max absolute empirical error:', (result.posterior - result.empirical_posterior).abs().max())
    print('Saved to:', output_dir.resolve())
    return result


def plot_results(result: pd.DataFrame, output_dir: Path, observations=OBSERVATIONS) -> None:
    # 延续项目已有的非交互式绘图与缓存位置。
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    colors = ('#d95f02', '#1b9e77', '#7570b3')
    for ax, personality in zip(axes.flat, Personality):
        subset = result[result.personality == personality.value]
        x = np.arange(1 + len(observations))
        for i, (strength, color) in enumerate(zip(STRENGTHS, colors)):
            records = subset[subset.strength == strength].set_index('observation').loc[list(observations)]
            heights = [records.prior.iloc[0], *records.posterior]
            ax.bar(x + (i - 1) * .25, heights, width=.25, color=color, label=strength)
            ax.scatter(x[1:] + (i - 1) * .25, records.empirical_posterior,
                       color='black', marker='_', s=50)
        ax.set_xticks(x, ['prior', *observations], rotation=20)
        ax.set_title(personality.value)
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
    axes.flat[0].legend()
    fig.suptitle('6 players, observer Ac Kc; bars: Bayes, black marks: simulation')
    fig.tight_layout()
    fig.savefig(output_dir / 'prior_posterior.png', dpi=180)
    plt.close(fig)


def add_context_arguments(parser):
    parser.add_argument('--pot', type=int, default=3)
    parser.add_argument('--current-bet', type=int, default=2)
    parser.add_argument('--street-bet', type=int, default=0)
    parser.add_argument('--min-raise', type=int, default=2)
    parser.add_argument('--big-blind', type=int, default=2)


def context_from_args(args):
    return BettingContext(args.pot, args.current_bet, args.street_bet, args.min_raise, args.big_blind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=261)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/bayes_6_players'))
    add_context_arguments(parser)
    args = parser.parse_args()
    run_experiment(args.trials, args.seed, args.output_dir, context_from_args(args))
