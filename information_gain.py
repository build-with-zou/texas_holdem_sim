"""精确计算已知性格时，动作与金额对底牌等级提供的信息（bit）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bayes_preflop import OBSERVATIONS, STRENGTHS, likelihoods, prior_from_cards, add_context_arguments, context_from_args
from personalities import Personality, PreflopTable, BettingContext, decision_likelihoods


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log2(positive)))


def observation_information(prior: np.ndarray, model: np.ndarray):
    """返回平均信息增益和各观察的概率、后验、熵、KL散度。

    KL衡量单个观察使分布变化多少；熵差衡量不确定性的变化，可为负。
    两者按观察概率取期望后均等于互信息。
    """
    prior = np.asarray(prior, dtype=float)
    model = np.asarray(model, dtype=float)
    if (prior.ndim != 1 or model.ndim != 2 or model.shape[0] != len(prior)
            or not np.isfinite(prior).all() or not np.isfinite(model).all()
            or (prior < 0).any() or (model < 0).any()
            or not np.isclose(prior.sum(), 1) or not np.allclose(model.sum(axis=1), 1)):
        raise ValueError('先验与似然须为维度匹配的有效概率分布。')
    joint = prior[:, None] * model
    evidence = joint.sum(axis=0)
    observations = []
    for j, probability in enumerate(evidence):
        if probability == 0:
            observations.append((0., None, None, None))
            continue
        posterior = joint[:, j] / probability
        mask = posterior > 0
        kl = float(np.sum(posterior[mask] * np.log2(posterior[mask] / prior[mask])))
        observations.append((float(probability), posterior, entropy(posterior), kl))
    gain = sum(p * kl for p, _, _, kl in observations if p > 0)
    return gain, observations


def run_analysis(players: int = 6, hero_cards: tuple[int, int] = (12, 11),
                 output_dir: Path = Path('outputs/information_6_players'),
                 context: BettingContext = BettingContext()) -> pd.DataFrame:
    observations = context.observations
    prior, _, _, _ = prior_from_cards(PreflopTable(), hero_cards, players)
    prior_entropy = entropy(prior)
    summaries, details = [], []
    for personality in Personality:
        full_model = likelihoods(personality, context)
        action_model = np.column_stack((full_model[:, :2], full_model[:, 2:].sum(axis=1)))
        action_gain, action_observations = observation_information(prior, action_model)
        full_gain, full_observations = observation_information(prior, full_model)
        raise_probability, raise_prior, _, _ = action_observations[2]
        # 已知加注时，金额在强/中/弱分布上进一步提供的信息。
        size_model = full_model[:, 2:] / full_model[:, 2:].sum(axis=1, keepdims=True)
        size_gain, _ = observation_information(raise_prior, size_model)
        summaries.append(dict(personality=personality.value, players=players,
                              hero_card_1=hero_cards[0], hero_card_2=hero_cards[1],
                              pot=context.pot, to_call=context.to_call,
                              current_bet=context.current_bet, street_bet=context.street_bet,
                              min_raise=context.min_raise, big_blind=context.big_blind,
                              prior_entropy_bits=prior_entropy,
                              action_information_bits=action_gain,
                              amount_extra_information_bits=full_gain - action_gain,
                              total_information_bits=full_gain,
                              remaining_entropy_bits=prior_entropy - full_gain,
                              uncertainty_reduction_fraction=full_gain / prior_entropy,
                              raise_probability=raise_probability,
                              amount_information_given_raise_bits=size_gain))
        for level, names, observations in (
            ('action', ('fold', 'call' if context.to_call else 'check',
                         'raise' if context.current_bet else 'bet'), action_observations),
            ('action_and_amount', observations, full_observations),
        ):
            for name, (probability, posterior, posterior_entropy, kl) in zip(names, observations):
                if probability == 0:
                    continue
                row = dict(personality=personality.value, level=level, observation=name,
                           probability=probability, posterior_entropy_bits=posterior_entropy,
                           entropy_reduction_bits=prior_entropy - posterior_entropy,
                           kl_from_prior_bits=kl)
                for i, strength in enumerate(STRENGTHS):
                    row[f'prior_{strength}'] = prior[i]
                    row[f'posterior_{strength}'] = posterior[i]
                details.append(row)
    summary = pd.DataFrame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / 'information_summary.csv', index=False)
    pd.DataFrame(details).to_csv(output_dir / 'observation_details.csv', index=False)
    plot_information(summary, output_dir)
    print(summary.to_string(index=False))
    return summary


def plot_information(summary: pd.DataFrame, output_dir: Path) -> None:
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    actions = summary.action_information_bits
    amounts = summary.amount_extra_information_bits
    ax.bar(x, actions, label='Action information', color='#1b9e77')
    ax.bar(x, amounts, bottom=actions, label='Extra amount information', color='#d95f02')
    for i, total in enumerate(summary.total_information_bits):
        ax.text(i, total + .004, f'{total:.4f}', ha='center')
    ax.set_xticks(x, summary.personality, rotation=12)
    ax.set_ylabel('Expected information gain (bits)')
    ax.set_title(f'Known personality: {int(summary.players.iloc[0])} players, observer card IDs '
                 f'{int(summary.hero_card_1.iloc[0])}, {int(summary.hero_card_2.iloc[0])}')
    ax.set_ylim(0, summary.total_information_bits.max() * 1.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'information_gain.png', dpi=180)
    plt.close(fig)


def run_price_comparison(output_dir: Path = Path('outputs/price_comparison')) -> None:
    """固定观察者先验，只改变公开下注条件，比较模型响应。"""
    contexts = {
        'baseline': BettingContext(),
        'raise_to_10': BettingContext(13, 10, 0, 8),
        'raise_to_40': BettingContext(43, 40, 0, 38),
    }
    rows, summaries = [], []
    for scenario, context in contexts.items():
        summary = run_analysis(output_dir=output_dir / scenario, context=context)
        summary.insert(0, 'scenario', scenario)
        summaries.append(summary)
        for personality in Personality:
            names, model = decision_likelihoods(personality, context)
            for i, strength in enumerate(STRENGTHS):
                rows.append(dict(scenario=scenario, personality=personality.value,
                                 strength=strength, pot=context.pot, to_call=context.to_call,
                                 call_price=context.call_price, min_raise=context.min_raise,
                                 raise_targets='/'.join(map(str, context.raise_targets)),
                                 fold_probability=model[i, 0], call_probability=model[i, 1],
                                 raise_probability=model[i, 2:].sum()))
    pd.concat(summaries, ignore_index=True).to_csv(output_dir / 'information_by_price.csv', index=False)
    pd.DataFrame(rows).to_csv(output_dir / 'action_probabilities_by_price.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--hero-cards', type=int, nargs=2, default=(12, 11))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/information_6_players'))
    parser.add_argument('--compare-prices', action='store_true', help='运行预设六人桌价格对照，保存到 outputs/price_comparison')
    add_context_arguments(parser)
    args = parser.parse_args()
    if args.compare_prices:
        run_price_comparison()
    else:
        run_analysis(args.players, tuple(args.hero_cards), args.output_dir, context_from_args(args))
