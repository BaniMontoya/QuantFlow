# Self-Learning FTMO-Ready EA (MQL5)

## Overview
This Expert Advisor is designed to be a "Self-Learning" agent that adapts to market conditions in real-time. Unlike static models, it uses Reinforcement Learning (RL) concepts implemented natively in MQL5 using the `matrix` and `vector` classes.

## Key Features
1. **Online Learning**: The model updates its weights after every trade (experience replay) or periodically based on new market data.
2. **FTMO Guard**: Built-in risk management that monitors:
   - Daily Drawdown (Default: < 5%)
   - Maximum Drawdown (Default: < 10%)
   - Profit Target
3. **Adaptive TP/SL**: The agent learns not just "when" to trade, but "how" to manage the trade (TakeProfit and StopLoss optimization).
4. **State Encoding**: 
   - Market Regime (Trend/Range)
   - Volatility (ATR)
   - Momentum (RSI/MACD)
   - Equity Health
5. **Ensemble Logic**: Can run multiple instances of the logic on different symbols, sharing a common risk manager.

## Architecture
- `RL_Agent.mqh`: Core learning logic (Q-Learning or Policy Gradient using MQL5 matrices).
- `RiskManager.mqh`: Global risk and drawdown protection.
- `FeatureFactory.mqh`: Normalizes market data into a state vector for the model.
- `SelfLearningEA.mq5`: The main entry point.

## Training Process
- **Cold Start**: Initial training on historical data (Strategy Tester).
- **Online Adaptation**: In real-time, the model adjusts its weights based on the "Reward" (Profit - Drawdown).
