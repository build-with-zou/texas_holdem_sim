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
- `winning_hand_distribution_[N]_players.png`：最大人数场景下的获胜牌型分布图。

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
