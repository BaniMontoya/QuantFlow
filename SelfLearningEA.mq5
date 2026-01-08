//+------------------------------------------------------------------+
//|                                              SelfLearningEA.mq5 |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include "RiskManager.mqh"
#include "RL_Agent.mqh"
#include "FeatureFactory.mqh"
#include <Trade\Trade.mqh>

//--- Input parameters
input double   InpMaxDailyDD = 4.5;      // Max Daily Drawdown (%)
input double   InpMaxTotalDD = 9.0;      // Max Total Drawdown (%)
input double   InpRiskPerTrade = 0.5;    // Risk per trade (%)
input int      InpStopLoss = 300;        // Initial SL (points)
input int      InpTakeProfit = 600;      // Initial TP (points)
input bool     InpTrainingMode = true;   // Training Mode (Exploration)

//--- Global Objects
CRiskManager   G_RiskManager;
CRLAgent       G_SignalAgent("Signal", 10, 24, 3);   // Direction: Neutral, Long, Short
CRLAgent       G_ManageAgent("Manage", 10, 16, 3);   // Management: Standard, Aggressive RR, Conservative RR
CTrade         G_Trade;

//--- State tracking
vector         G_LastState;
int            G_LastSigAction = 0;
int            G_LastManAction = 0;
bool           G_TradeActive = false;


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    G_RiskManager.SetLimits(InpMaxDailyDD, InpMaxTotalDD);
    G_RiskManager.OnNewDay();
    
    if (!InpTrainingMode)
    {
        G_SignalAgent.SetEpsilon(0.01);
        G_ManageAgent.SetEpsilon(0.01);
    }
    else
    {
        G_SignalAgent.SetEpsilon(0.2);
        G_ManageAgent.SetEpsilon(0.15);
    }
        
    Print("Self-Learning EA Initialized. Mode: ", InpTrainingMode ? "TRAINING" : "LIVE");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // 1. Mandatory Risk Check
    if (!G_RiskManager.IsTradingAllowed())
        return;

    // 2. Check for Day change to reset daily DD
    static datetime last_day = 0;
    datetime current_day = iTime(_Symbol, PERIOD_D1, 0);
    if (current_day != last_day)
    {
        G_RiskManager.OnNewDay();
        last_day = current_day;
    }

    // 3. Trade Management & Learning
    CheckActiveTrades();

    // 4. Decision Logic
    if (!G_TradeActive)
    {
        vector current_state = CFeatureFactory::GetState(_Symbol, _Period);
        int sig_action = G_SignalAgent.GetAction(current_state);
        
        if (sig_action == 0) return; // Neutral
        
        int man_action = G_ManageAgent.GetAction(current_state);
        
        double tp_mult = 1.0, sl_mult = 1.0;
        if(man_action == 1) { tp_mult = 2.0; sl_mult = 0.5; } // Aggressive
        if(man_action == 2) { tp_mult = 0.5; sl_mult = 1.5; } // Conservative
        
        if (sig_action == 1) // Buy
        {
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - (InpStopLoss * sl_mult) * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + (InpTakeProfit * tp_mult) * _Point;
            double lots = G_RiskManager.GetMaxLotSize(InpRiskPerTrade, InpStopLoss * sl_mult);
            
            if (G_Trade.Buy(lots, _Symbol, 0, sl, tp, "RL_SIGNAL_BUY"))
            {
                G_LastState = current_state;
                G_LastSigAction = 1;
                G_LastManAction = man_action;
                G_TradeActive = true;
            }
        }
        else if (sig_action == 2) // Sell
        {
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_BID) + (InpStopLoss * sl_mult) * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_BID) - (InpTakeProfit * tp_mult) * _Point;
            double lots = G_RiskManager.GetMaxLotSize(InpRiskPerTrade, InpStopLoss * sl_mult);
            
            if (G_Trade.Sell(lots, _Symbol, 0, sl, tp, "RL_SIGNAL_SELL"))
            {
                G_LastState = current_state;
                G_LastSigAction = 2;
                G_LastManAction = man_action;
                G_TradeActive = true;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check and handle closed trades for learning                      |
//+------------------------------------------------------------------+
void CheckActiveTrades()
{
    if (!G_TradeActive) return;

    if (!PositionSelectByMagic(0)) 
    {
        HistorySelect(TimeCurrent() - 3600, TimeCurrent());
        uint total = HistoryDealsTotal();
        if (total > 0)
        {
            ulong ticket = HistoryDealGetTicket(total - 1);
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + HistoryDealGetDouble(ticket, DEAL_SWAP) + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
            
            double reward = profit; 
            vector next_state = CFeatureFactory::GetState(_Symbol, _Period);
            
            // Train BOTH agents
            G_SignalAgent.Train(G_LastState, G_LastSigAction, reward, next_state, true);
            G_ManageAgent.Train(G_LastState, G_LastManAction, reward, next_state, true);
            
            Print("Reward: ", reward, ". SigAction: ", G_LastSigAction, " ManAction: ", G_LastManAction);
        }
        G_TradeActive = false;
    }
}


//+------------------------------------------------------------------+
//| Helper to select position                                        |
//+------------------------------------------------------------------+
bool PositionSelectByMagic(long magic)
{
    for(int i=PositionsTotal()-1; i>=0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            // In this simplified version, we just check if any position is open
            return true;
        }
    }
    return false;
}
