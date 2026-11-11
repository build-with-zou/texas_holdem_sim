import unittest
from dataclasses import fields

import numpy as np

from personalities import Personality, PreflopTable
from poker_game import Action, Street, create_game
from preflop_hand import (PreflopHand, play_hand, random_decision, personality_decision,
                          personality_distribution, data_driven_decision,
                          opponent_strength_posteriors)


class MinimalHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = PreflopTable()

    def hand(self, players=6, button=0, max_raises=3, deck=None):
        return PreflopHand([Personality.LOOSE_AGGRESSIVE] * players,
                           np.random.default_rng(261), button, max_raises, deck)

    def call(self, hand):
        legal = dict(hand.view().legal_actions)
        action = Action.CHECK if Action.CHECK in legal else Action.CALL
        hand.apply(action, legal[action])

    def test_big_blind_keeps_option(self):
        hand = self.hand(3)
        self.call(hand)
        self.call(hand)
        self.assertFalse(hand.complete)
        self.assertEqual(hand.view().player_id, 2)
        self.assertIn(Action.CHECK, dict(hand.view().legal_actions))
        self.assertIn(Action.RAISE, dict(hand.view().legal_actions))
        self.call(hand)
        self.assertTrue(hand.complete)

    def test_new_street_needs_actions(self):
        game = create_game(3)
        for action in (Action.CALL, Action.CALL, Action.CHECK):
            game.apply_action(action)
        game.advance_street([0, 1, 2])
        self.assertEqual(game.street, Street.FLOP)
        self.assertFalse(game.betting_round_complete())
        for _ in range(3):
            game.apply_action(Action.CHECK)
        self.assertTrue(game.betting_round_complete())

    def test_raise_reopens_action_and_call_cost(self):
        hand = self.hand(3)
        hand.apply(Action.CALL, 2)
        hand.apply(Action.CALL, 1)
        hand.apply(Action.RAISE, 8)  # BB总投入从2到10，本次追加8
        self.assertFalse(hand.complete)
        self.assertEqual(hand.view().player_id, 0)
        self.assertEqual(hand.view().context.to_call, 8)
        self.call(hand)
        self.assertFalse(hand.complete)
        self.call(hand)
        result = hand.settle()
        self.assertEqual(result['gross_pot'], 30)
        self.assertEqual([p['committed'] for p in result['players']], [10, 10, 10])
        self.assertEqual(sum(p['net_profit'] for p in result['players']), 0)

    def test_fold_win_refunds_uncalled_raise(self):
        hand = self.hand()
        hand.apply(Action.RAISE, 10)  # UTG座位3
        for _ in range(5):
            hand.apply(Action.FOLD, 0)
        result = hand.settle()
        self.assertEqual(result['winners'], [3])
        self.assertEqual(result['reason'], 'all_others_folded')
        self.assertEqual(result['gross_pot'], 13)
        self.assertEqual(result['uncalled_refund'], 8)
        self.assertEqual(result['contested_pot'], 5)
        self.assertEqual(result['players'][3]['net_profit'], 3)
        self.assertEqual(result['community_cards'], [])
        with self.assertRaises(ValueError):
            hand.settle()
        with self.assertRaises(ValueError):
            hand.apply(Action.CHECK)

    def test_raise_cap_preserves_call_and_normalized_strategy(self):
        hand = self.hand(3, max_raises=1)
        hand.apply(Action.RAISE, 10)
        view = hand.view()
        self.assertEqual(view.raises_remaining, 0)
        self.assertNotIn(Action.RAISE, dict(view.legal_actions))
        self.assertIn(Action.CALL, dict(view.legal_actions))
        dist = personality_distribution(view, self.table)
        self.assertAlmostEqual(sum(dist.values()), 1)
        self.assertTrue(all(a not in (Action.RAISE, Action.ALL_IN) for a, _ in dist))
        with self.assertRaises(ValueError):
            hand.apply(Action.RAISE, 20)
        self.call(hand)
        self.call(hand)
        self.assertTrue(hand.complete)

    def test_heads_up_blinds_and_order(self):
        for button in range(2):
            hand = self.hand(2, button)
            view = hand.view()
            self.assertEqual(view.player_id, button)
            self.assertEqual(view.players[button].position, 'BTN/SB')
            self.assertEqual(view.context.to_call, 1)
            self.call(hand)
            self.assertEqual(hand.view().player_id, 1 - button)
            self.call(hand)
            self.assertTrue(hand.complete)

    def test_showdown_tie_and_odd_chip(self):
        # 公共牌为Tc Jc Qc Kc Ac，存活的座位0与2打平，座位1弃牌。
        first = [13, 14, 15, 16, 17, 18, 8, 9, 10, 11, 12]
        deck = first + [c for c in range(52) if c not in first]
        hand = self.hand(3, deck=deck)
        self.call(hand)
        hand.apply(Action.FOLD, 0)
        self.call(hand)
        result = hand.settle()
        self.assertEqual(result['winners'], [0, 2])
        self.assertEqual(result['contested_pot'], 5)
        self.assertEqual([p['award'] for p in result['players']], [2, 0, 3])
        self.assertEqual([p['net_profit'] for p in result['players']], [0, -1, 1])

    def test_showdown_unique_winner(self):
        # 座位0有两张黑桃，与公共牌组成A高同花，其余玩家不成同花。
        first = [51, 50, 0, 14, 28, 3, 48, 46, 44, 6, 21]
        deck = first + [c for c in range(52) if c not in first]
        hand = self.hand(3, deck=deck)
        while not hand.complete:
            self.call(hand)
        result = hand.settle()
        self.assertEqual(result['winners'], [0])
        self.assertEqual([p['net_profit'] for p in result['players']], [4, -2, -2])

    def test_strategy_view_hides_opponent_cards_and_runout(self):
        hand = self.hand()
        view = hand.view()
        self.assertEqual(len(view.hole_cards), 2)
        self.assertEqual(view.community_cards, ())
        self.assertNotIn('hole_cards', {f.name for f in fields(view.players[0])})
        self.assertNotIn('deck', {f.name for f in fields(view)})
        with self.assertRaises(AttributeError):
            view.players[0].committed = 100
        with self.assertRaises(ValueError):
            hand.settle()

    def test_data_driven_strategy_uses_public_history_and_hides_cards(self):
        hand = self.hand(6)
        first = hand.view()
        self.assertEqual(len(opponent_strength_posteriors(first, self.table)), 5)
        # 记录一个公开弃牌后，策略仍只接收行动历史和自己的底牌。
        hand.apply(Action.FOLD, 0)
        view = hand.view()
        posteriors = opponent_strength_posteriors(view, self.table)
        self.assertNotIn(first.player_id, posteriors)
        action, amount = data_driven_decision(view, np.random.default_rng(1), self.table)
        self.assertIn(action, dict(view.legal_actions))
        if action in (Action.BET, Action.RAISE):
            self.assertGreaterEqual(amount, dict(view.legal_actions)[action])

    def test_data_driven_hand_runs_and_logs_policy(self):
        streams = np.random.SeedSequence(77).spawn(7)
        hand = self.hand(6)
        table = self.table
        def policy(view, rng):
            return data_driven_decision(view, rng, table)
        result = play_hand(hand, [policy] * 6, [np.random.default_rng(s) for s in streams[1:]])
        self.assertEqual(sum(p['net_profit'] for p in result['players']), 0)
        self.assertIn(result['reason'], {'showdown', 'all_others_folded'})

    def test_invalid_deal_and_action(self):
        with self.assertRaises(ValueError):
            self.hand(deck=[0] * 52)
        with self.assertRaises(ValueError):
            self.hand(max_raises=-1)
        hand = self.hand()
        with self.assertRaises(ValueError):
            hand.apply(Action.CHECK)
        with self.assertRaises(ValueError):
            hand.apply(Action.CALL, 20)
        with self.assertRaises(ValueError):
            hand.apply(Action.RAISE, 3)

    def test_same_deal_across_different_policies(self):
        results = []
        boards = []
        for policy in ('call', 'random'):
            streams = np.random.SeedSequence(261).spawn(7)
            hand = PreflopHand([Personality.LOOSE_AGGRESSIVE] * 6, np.random.default_rng(streams[0]))
            def call_policy(view, rng):
                legal = dict(view.legal_actions)
                action = Action.CALL if Action.CALL in legal else Action.CHECK
                return action, legal[action]
            boards.append(hand._board)
            policies = [call_policy if policy == 'call' else random_decision] * 6
            results.append(play_hand(hand, policies, [np.random.default_rng(s) for s in streams[1:]]))
        self.assertEqual(boards[0], boards[1])
        self.assertEqual([p['hole_cards'] for p in results[0]['players']],
                         [p['hole_cards'] for p in results[1]['players']])

    def test_reproducible_bounded_zero_sum_games(self):
        def run(seed, n, cap):
            streams = np.random.SeedSequence(seed).spawn(n + 1)
            hand = self.hand(n, button=seed % n, max_raises=cap)
            policies = [random_decision if i % 2 else
                        lambda view, rng: personality_decision(view, rng, self.table) for i in range(n)]
            return play_hand(hand, policies, [np.random.default_rng(s) for s in streams[1:]])
        for n in (2, 3, 6, 10):
            for cap in (0, 1, 3):
                for seed in range(10):
                    result = run(seed, n, cap)
                    self.assertEqual(sum(p['net_profit'] for p in result['players']), 0)
                    self.assertLessEqual(len(result['actions']), n * (cap + 1))
                    self.assertLessEqual(result['raises_used'], cap)
                    cards = [c for p in result['players'] for c in p['hole_cards']] + result['community_cards']
                    self.assertEqual(len(cards), len(set(cards)))
        self.assertEqual(run(261, 6, 3), run(261, 6, 3))
