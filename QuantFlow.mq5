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
#include "GeneticOptimizer.mqh"
#include <Trade\Trade.mqh>

//--- Input parameters
input double   InpMaxDailyDD = 4.5;      // Max Daily Drawdown (%)
input double   InpMaxTotalDD = 9.0;      // Max Total Drawdown (%)
input double   InpRiskPerTrade = 0.5;    // Risk per trade (%)
input int      InpStopLoss = 300;        // Initial SL (points)
input int      InpTakeProfit = 600;      // Initial TP (points)
input bool     InpTrainingMode = true;   // Training Mode (Exploration)

//--- Global Objects
CRiskManager       G_RiskManager;
CRLAgent           G_SignalAgent("Signal", 10, 24, 3);   // Direction: Neutral, Long, Short
CRLAgent           G_ManageAgent("Manage", 10, 16, 3);   // Management: Standard, Aggressive RR, Conservative RR
CGeneticOptimizer  G_GAOptimizer(30);
CTrade             G_Trade;

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
    
    G_GAOptimizer.Initialize();
    G_GAOptimizer.Evolve();
    
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
        
    Print("Self-Learning EA Initialized with Internal GA. Mode: ", InpTrainingMode ? "TRAINING" : "LIVE");
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

    // 4. Genetic Optimization Update (every 4 hours)
    static datetime last_ga_update = 0;
    if (TimeCurrent() - last_ga_update > 4 * 3600)
    {
        G_GAOptimizer.Evolve();
        last_ga_update = TimeCurrent();
    }

    // 5. Decision Logic
    if (!G_TradeActive)
    {
        vector current_state = CFeatureFactory::GetState(_Symbol, _Period);
        int sig_action = G_SignalAgent.GetAction(current_state);
        
        if (sig_action == 0) return; // Neutral
        
        int man_action = G_ManageAgent.GetAction(current_state);
        
        // Obtenemos los mejores SL/TP del optimizador genético
        int opt_sl = 0, opt_tp = 0;
        G_GAOptimizer.GetBestParameters(opt_sl, opt_tp);
        
        double tp_mult = 1.0, sl_mult = 1.0;
        if(man_action == 1) { tp_mult = 1.5; sl_mult = 0.8; } // Ajustado por RL
        if(man_action == 2) { tp_mult = 0.8; sl_mult = 1.2; } // Ajustado por RL
        
        double final_sl = opt_sl * sl_mult;
        double final_tp = opt_tp * tp_mult;
        
        if (sig_action == 1) // Buy
        {
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - (final_sl) * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + (final_tp) * _Point;
            double lots = G_RiskManager.GetMaxLotSize(InpRiskPerTrade, final_sl);
            
            if (G_Trade.Buy(lots, _Symbol, 0, sl, tp, "RL_GA_BUY"))
            {
                G_LastState = current_state;
                G_LastSigAction = 1;
                G_LastManAction = man_action;
                G_TradeActive = true;
            }
        }
        else if (sig_action == 2) // Sell
        {
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_BID) + (final_sl) * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_BID) - (final_tp) * _Point;
            double lots = G_RiskManager.GetMaxLotSize(InpRiskPerTrade, final_sl);
            
            if (G_Trade.Sell(lots, _Symbol, 0, sl, tp, "RL_GA_SELL"))
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
