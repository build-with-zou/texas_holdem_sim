import unittest
import numpy as np

from personalities import (BettingContext, Personality, PreflopTable,
                           decision_likelihoods, choose_decision, action_probabilities)
from poker_game import Action, create_game
from bayes_preflop import update_posterior
from information_gain import observation_information


class BettingPriceTests(unittest.TestCase):
    def test_higher_call_price_reduces_continuation(self):
        contexts = [BettingContext(), BettingContext(11, 10, 0, 8), BettingContext(41, 40, 0, 38)]
        for personality in Personality:
            models = [decision_likelihoods(personality, c)[1] for c in contexts]
            for low, high in zip(models, models[1:]):
                self.assertTrue(np.all(high[:, 0] > low[:, 0]))
                self.assertTrue(np.all(high[:, 2:].sum(axis=1) < low[:, 2:].sum(axis=1)))

    def test_same_amount_is_cheaper_in_larger_pot(self):
        for personality in Personality:
            small = decision_likelihoods(personality, BettingContext(11, 10, 0, 8))[1]
            large = decision_likelihoods(personality, BettingContext(51, 10, 0, 8))[1]
            self.assertTrue(np.all(large[:, 0] <= small[:, 0]))

    def test_expensive_minimum_raise_changes_decision(self):
        for personality in Personality:
            cheap = decision_likelihoods(personality, BettingContext(21, 20, 0, 2))[1]
            expensive = decision_likelihoods(personality, BettingContext(21, 20, 0, 20))[1]
            self.assertTrue(np.all(expensive[:, 2:].sum(axis=1) < cheap[:, 2:].sum(axis=1)))
            np.testing.assert_allclose(expensive[:, 0], cheap[:, 0])

    def test_scale_invariance_and_incremental_cost(self):
        for personality in Personality:
            _, a = decision_likelihoods(personality)
            _, b = decision_likelihoods(personality, BettingContext(30, 20, 0, 20, 20))
            np.testing.assert_allclose(a, b)
        self.assertEqual(BettingContext(30, 20, 15, 10).to_call, 5)

    def test_merged_amounts_free_check_and_posterior(self):
        context = BettingContext(3, 2, 0, 10)
        # 退化档位必须合并，防止金额之外泄漏不可见的档位信息。
        names, model = decision_likelihoods(Personality.TIGHT_AGGRESSIVE, context)
        self.assertEqual(names, ('fold', 'call_2', 'raise_to_12'))
        np.testing.assert_allclose(model.sum(axis=1), 1)
        self.assertAlmostEqual(update_posterior(np.array([.1, .6, .3]), model, 2).sum(), 1)
        names, model = decision_likelihoods(Personality.TIGHT_AGGRESSIVE, BettingContext(10, 2, 2, 2))
        self.assertEqual(names[1], 'check')
        np.testing.assert_array_equal(model[:, 0], 0)
        self.assertTrue(np.isfinite(observation_information(np.array([.1, .6, .3]), model)[0]))

    def test_sampled_amounts_are_applied_and_match_distribution(self):
        table = PreflopTable()
        rng = np.random.default_rng(261)
        g = create_game(6, starting_stack=100)
        p = g.players[g.to_act]
        p.hole_cards = (12, 25)
        p.personality = Personality.TIGHT_AGGRESSIVE
        samples = [choose_decision(g, table, rng) for _ in range(5000)]
        self.assertEqual({amount for action, amount in samples if action == Action.RAISE}, {4, 6, 10})
        names, model = decision_likelihoods(p.personality)
        for j, amount in enumerate((4, 6, 10), start=2):
            rate = samples.count((Action.RAISE, amount)) / len(samples)
            self.assertAlmostEqual(rate, model[0, j], delta=.025)
        action, amount = next(s for s in samples if s == (Action.RAISE, 10))
        before = g.pot + sum(p.stack for p in g.players)
        g.apply_action(action, amount)
        self.assertEqual(g.current_bet, 10)
        self.assertEqual(g.min_raise, 8)
        self.assertEqual(g.pot + sum(p.stack for p in g.players), before)

    def test_actual_game_context_uses_price(self):
        table = PreflopTable()
        g = create_game(6)
        for p in g.players:
            p.hole_cards = (12, 25)
            p.personality = Personality.TIGHT_AGGRESSIVE
        g.apply_action(Action.RAISE, 10)
        names, expected = decision_likelihoods(g.players[g.to_act].personality, BettingContext(13, 10, 0, 8))
        self.assertAlmostEqual(action_probabilities(g, table)[Action.FOLD], expected[0, 0])

    def test_invalid_context(self):
        for kwargs in ({'pot': -1}, {'min_raise': 0}, {'pot': float('nan')}, {'street_bet': 3}):
            with self.assertRaises(ValueError):
                BettingContext(**kwargs)
