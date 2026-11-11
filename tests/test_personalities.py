import unittest
import numpy as np

from personalities import (Personality, PreflopTable, action_probabilities,
                           choose_action, hand_class)
from poker_game import Action, Street, create_game


class PersonalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = PreflopTable()

    def test_hand_classes_and_suit_symmetry(self):
        self.assertEqual(hand_class((12, 25)), 'AA')
        self.assertEqual(hand_class((12, 11)), 'AKs')
        self.assertEqual(hand_class((24, 12)), 'AKo')
        self.assertEqual(hand_class((38, 37)), 'AKs')
        with self.assertRaises(ValueError):
            hand_class((12, 12))

    def test_lookup_all_player_counts(self):
        for n in range(2, 11):
            self.assertEqual(self.table.lookup((12, 25), n).strength, 'strong')
        with self.assertRaises(ValueError):
            self.table.lookup((12, 25), 11)

    def test_expected_probabilities(self):
        g = create_game(6)
        g.players[g.to_act].hole_cards = (12, 25)
        g.players[g.to_act].personality = Personality.TIGHT_AGGRESSIVE
        p = action_probabilities(g, self.table)
        self.assertAlmostEqual(p[Action.FOLD], .05)
        self.assertAlmostEqual(p[Action.RAISE], .95 * .8)
        self.assertAlmostEqual(p[Action.CALL], .95 * .2)

    def test_free_check_and_short_stack(self):
        g = create_game(6)
        g.players[g.to_act].hole_cards = (0, 14)
        g.players[g.to_act].street_bet = g.current_bet
        self.assertEqual(action_probabilities(g, self.table)[Action.FOLD], 0)
        g.players[g.to_act].street_bet = 0
        g.players[g.to_act].stack = 1
        p = action_probabilities(g, self.table)
        self.assertNotIn(Action.RAISE, p)
        self.assertAlmostEqual(sum(p.values()), 1)
        g.street = Street.FLOP
        with self.assertRaises(ValueError):
            action_probabilities(g, self.table)

    def test_sampling_and_game_integration(self):
        rng = np.random.default_rng(261)
        g = create_game(6)
        g.players[g.to_act].hole_cards = (12, 25)
        g.players[g.to_act].personality = Personality.TIGHT_AGGRESSIVE
        samples = [choose_action(g, self.table, rng) for _ in range(10000)]
        self.assertAlmostEqual(samples.count(Action.RAISE) / len(samples), .76, delta=.02)
        before = g.pot + sum(p.stack for p in g.players)
        action = choose_action(g, self.table, rng)
        self.assertIn(action, g.legal_actions())
        g.apply_action(action)
        self.assertEqual(g.pot + sum(p.stack for p in g.players), before)


if __name__ == '__main__':
    unittest.main()
