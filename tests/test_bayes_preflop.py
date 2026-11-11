import unittest
import numpy as np
from bayes_preflop import (likelihoods, prior_from_cards, update_posterior,
                           Personality, PreflopTable)


class BayesTests(unittest.TestCase):
    def test_prior_excludes_observer_cards(self):
        prior, counts, hands, labels = prior_from_cards(PreflopTable(), (12, 11))
        self.assertEqual(len(hands), 1225)
        self.assertEqual(int(counts.sum()), 1225)
        self.assertTrue(all(12 not in hand and 11 not in hand for hand in hands))
        np.testing.assert_array_equal(counts, np.bincount(labels, minlength=3))
        self.assertAlmostEqual(prior.sum(), 1)

    def test_bayes_against_hand_calculation(self):
        prior = np.array([.1, .5, .4])
        model = np.array([[.6, .1, .1, .1, .1],
                          [.1, .6, .1, .1, .1],
                          [.1, .6, .1, .1, .1]])
        self.assertAlmostEqual(update_posterior(prior, model, 0)[0], .4)
        # 当观察对三类牌没有区分能力时，后验等于先验。
        np.testing.assert_allclose(update_posterior(prior, model, 2), prior)

    def test_models_and_total_probability(self):
        prior = np.array([.1, .6, .3])
        for personality in Personality:
            model = likelihoods(personality)
            np.testing.assert_allclose(model.sum(axis=1), 1)
            evidence = prior @ model
            posteriors = np.array([update_posterior(prior, model, i) for i in range(5)])
            np.testing.assert_allclose(posteriors.sum(axis=1), 1)
            np.testing.assert_allclose(evidence @ posteriors, prior)

    def test_impossible_observation(self):
        model = np.zeros((3, 5))
        model[:, 0] = 1
        with self.assertRaises(ValueError):
            update_posterior(np.array([.1, .6, .3]), model, 1)
