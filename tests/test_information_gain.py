import unittest
import numpy as np
from information_gain import entropy, observation_information
from bayes_preflop import likelihoods, prior_from_cards
from personalities import Personality, PreflopTable


class InformationTests(unittest.TestCase):
    def test_independent_observation(self):
        prior = np.array([.1, .6, .3])
        model = np.tile([.2, .8], (3, 1))
        gain, _ = observation_information(prior, model)
        self.assertAlmostEqual(gain, 0)

    def test_perfect_observation(self):
        prior = np.array([.1, .6, .3])
        gain, _ = observation_information(prior, np.eye(3))
        self.assertAlmostEqual(gain, entropy(prior))

    def test_zero_probability_observation(self):
        gain, rows = observation_information(np.array([1., 0., 0.]), np.eye(3))
        self.assertEqual(gain, 0)
        self.assertIsNone(rows[1][1])

    def test_chain_rule_and_entropy_identity(self):
        prior, _, _, _ = prior_from_cards(PreflopTable(), (12, 11))
        for personality in Personality:
            full = likelihoods(personality)
            action = np.column_stack((full[:, :2], full[:, 2:].sum(axis=1)))
            total, full_rows = observation_information(prior, full)
            action_gain, action_rows = observation_information(prior, action)
            p_raise, given_raise, _, _ = action_rows[2]
            size_gain, _ = observation_information(given_raise, full[:, 2:] / full[:, 2:].sum(axis=1, keepdims=True))
            self.assertAlmostEqual(total, action_gain + p_raise * size_gain)
            self.assertAlmostEqual(total, entropy(prior) - sum(p * h for p, _, h, _ in full_rows))
            self.assertGreaterEqual(total, action_gain)
            self.assertLessEqual(total, entropy(prior))

    def test_invalid_distribution(self):
        with self.assertRaises(ValueError):
            observation_information(np.array([.5, .6]), np.eye(2))
