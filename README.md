# 德州扑克纯概率模拟

这个目录是概率统计课程论文的代码实现。当前版本是**纯概率基线模型**：

- 每局使用一副 52 张牌；
- 每位玩家发 2 张底牌；
- 发出 5 张公共牌；
- 所有玩家都不弃牌、不下注、不诈唬，直接摊牌比较最大 5 张牌；
- 统计不同上桌人数下，固定 0 号玩家的胜率、独赢率、平局率，以及最终获胜牌型分布。

这个模型不是完整德州扑克策略模型，而是后续加入位置、玩家性格、下注规则之前的 baseline。

## 环境配置

在本目录下执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行模拟

默认对 2 到 10 人桌分别模拟 5000 局：

```bash
python simulate.py
```

也可以指定模拟次数和玩家人数：

```bash
python simulate.py --trials 50000 --players 2 6 9 --seed 261
```

## 输出文件

运行后会生成 `outputs/` 目录：

- `summary_by_player_count.csv`：不同人数下的胜率、独赢率、平局率；
- `winning_hand_distribution.csv`：不同人数下最终获胜牌型分布；
- `win_probability_by_player_count.png`：胜率随玩家人数变化图；
- `tie_rate_by_player_count.png`：平局率随玩家人数变化图；
- `winning_hand_distribution_[N]_players.png`：每种人数场景下的获胜牌型分布图；
- `winning_hand_distribution_comparison.png`：不同人数下获胜牌型分布的汇总对比图。

## 当前模型的解释边界

当前版本回答的问题是：

> 如果所有玩家都一直留到最后，随机发牌本身会导致怎样的胜率与牌型分布？

它暂时不回答：

- 按钮位、大盲位等位置优势；
- 玩家是否弃牌；
- 下注、加注、跟注；
- 玩家性格，如紧、松、激进、保守；
- 长期收益与筹码变化。

这些会作为下一阶段策略模型加入。

## 策略模型的基础实现

当前已新增 `poker_game.py`，为后续策略实验提供基础框架：

- `GameState` 保存按钮位、盲注、筹码、底池、公共牌、当前下注额和行动顺序；
- `PlayerState` 保存每位玩家的筹码、弃牌、全押和本轮/全局投入；
- 支持翻牌前、翻牌、转牌、河牌四个下注阶段；
- 支持弃牌、过牌、跟注、下注、加注和全押，并检查当前动作是否合法；
- 支持结束下注轮并推进到下一街。

该模块暂时不决定玩家应该采取什么动作，玩家策略和牌力估计将在下一步接入。

## 起手牌强度实验

使用 `preflop_equity.py` 可以在指定人数下计算 169 类标准起手牌的条件权益。例如 6 人桌：

```bash
.venv/bin/python preflop_equity.py --players 6 --trials 5000 --seed 261 \
  --output outputs/preflop_strength_6_players.csv
```

`relative_to_average` 是该起手牌权益与理论平均权益 `1 / 玩家人数` 的比值：比值至少为 1.5 判为强牌，0.75 到 1.5 判为中等牌，低于 0.75 判为弱牌。模拟次数越多，牌力排序越稳定。

## 翻牌前性格策略

`personalities.py` 提供 `PreflopTable.lookup`、`action_probabilities` 和
`choose_action`。玩家的 `hole_cards` 使用 `holdem.py` 的牌 ID，
`personality` 使用 `Personality` 枚举；默认紧弱，可为每位玩家分别设置。
查表使用本局初始人数（2–10），对手弃牌后仍沿用原人数，避免将有选择地
留下的对手误当成随机对手。牌力为权益乘人数，≥1.5 为强牌，≥0.75 为中等牌。

以下概率是人为设定的实验参数，不是拟合结果。元组顺序是强牌、中等牌、弱牌：

| 性格 | 面对跟注成本时继续概率 | 继续后主动下注/加注概率 |
|---|---|---|
| 紧弱 | 95%、20%、5% | 10%、5%、1% |
| 紧凶 | 95%、20%、5% | 80%、50%、20% |
| 松弱 | 99%、80%、40% | 10%、5%、1% |
| 松凶 | 99%、80%、40% | 80%、50%、20% |

每次行动重新抽样；继续且不主动加注时跟注。可以免费过牌时始终继续，
按第二组概率下注/加注，否则过牌；不能加注时将这部分概率归入跟注/过牌。
主动动作仅使用当前引擎的最小下注/加注，不额外随机全押，短筹码跟注可能全押。
目前不考虑跟注金额、位置、对手范围、历史行动，且拒绝在翻牌后使用起手牌表。

```python
import numpy as np
from personalities import Personality, PreflopTable, choose_action
from poker_game import create_game

game = create_game(6)
player = game.players[game.to_act]
player.hole_cards = (12, 25)  # Ac Ad
player.personality = Personality.TIGHT_AGGRESSIVE
rng = np.random.default_rng(261)
table = PreflopTable()
action = choose_action(game, table, rng)
game.apply_action(action)
```

验证：`.venv/bin/python -m unittest discover -s tests -v`。
这只是决策接口，完整下注轮结束判断、全押重开加注和边池结算仍需完善后才能进行收益实验。

## 六人桌贝叶斯首次行动实验

运行 `.venv/bin/python bayes_preflop.py --trials 100000 --seed 261`。
结果位于 `outputs/bayes_6_players/`：`prior.csv` 保存先验，
`posterior_comparison.csv` 保存观察似然、理论后验、模拟后验和样本数，
`prior_posterior.png` 对比先验与观察后的分布（柱为理论值、黑线为模拟频率）。

固定观察者底牌 Ac Kc，目标是六人桌翻牌前第一个行动者（非盲注位）。
底池3、目标跟注需2，目标可弃牌、跟注2、加注到4/6/10。
先枚举排除观察者底牌后的1225个对手底牌组合，按六人桌权益表归为强/中/弱。
其余四位玩家的隐藏牌边缘化，不需要发出；目标行动之前无其他行动信息。
本实验不设置筹码上限、不使用全押，只研究一次行动，不运行整局结算。

行为模型沿用 `POLICIES` 的继续概率和继续后的进攻概率。
选择进攻后，再按下表随机选择加注金额：

| 性格组 | 牌力 | 加注到4 | 加注到6 | 加注到10 |
|---|---|---|---|---|
| 紧弱、松弱 | 强 | 60% | 30% | 10% |
| 紧弱、松弱 | 中 | 75% | 20% | 5% |
| 紧弱、松弱 | 弱 | 85% | 13% | 2% |
| 紧凶、松凶 | 强 | 20% | 40% | 40% |
| 紧凶、松凶 | 中 | 40% | 40% | 20% |
| 紧凶、松凶 | 弱 | 55% | 30% | 15% |

例如 P(加注到10 | 强牌、紧凶) = 0.95 × 0.80 × 0.40。
`update_posterior` 只接收先验和观察似然，不接收真实底牌。
四种性格各独立抽样10万次，真实底牌仅用于统计观察后的经验牌力分布。
这是同一已知生成模型下的计算验证，不是现实玩家性格或策略有效性的实证；
金额概率是人为假设，牌力分类还包含原5000局权益表的抽样误差。
当前金额策略用于本独立实验，`poker_game.py` 的完整下注引擎尚未改为此金额规则。
