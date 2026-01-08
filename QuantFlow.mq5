//+------------------------------------------------------------------+
//|                                                   QuantFlow.mq5 |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property version   "2.00"
#property strict

#include "RiskManager.mqh"
#include "RL_Agent.mqh"
#include "FeatureFactory.mqh"
#include "GeneticOptimizer.mqh"
#include "ONNXWrapper.mqh"
#include <Trade\Trade.mqh>

//--- Input parameters (Rangos para el GA)
input group "=== Risk Management ==="
input double   InpMaxDailyDD = 4.5;      // Max Daily Drawdown (%)
input double   InpMaxTotalDD = 9.0;      // Max Total Drawdown (%)
input double   InpRiskPerTrade = 0.5;    // Risk per trade (%)

input group "=== GA Search Bounds ==="
input int      InpMinSL = 100;           // Min Stop Loss (Points)
input int      InpMaxSL = 1500;          // Max Stop Loss (Points)
input int      InpMinTP = 100;           // Min Take Profit (Points)
input int      InpMaxTP = 3000;          // Max Take Profit (Points)

input group "=== Training Settings ==="
input bool     InpTrainingMode = true;   // Live Training Mode

//--- Global Objects
CRiskManager       G_RiskManager;
CRLAgent           G_SignalAgent("Signal", 10, 32, 3);
CRLAgent           G_ManageAgent("Manage", 10, 24, 3);
CGeneticOptimizer  G_GAOptimizer(40);
CONNXModel         G_LSTMModel;
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
    
    // Configuramos el Optimizador Genético con los rangos elegidos
    G_GAOptimizer.SetBounds(InpMinSL, InpMaxSL, InpMinTP, InpMaxTP);
    G_GAOptimizer.Initialize();
    G_GAOptimizer.Evolve();
    
    // Intentar cargar modelo ONNX (si existe)
    // Nota: El usuario debe copiar "QuantFlow_LSTM.onnx" a MQL5/Files/
    if(G_LSTMModel.Load("QuantFlow_LSTM.onnx"))
    {
        Print("Modelo LSTM cargado y listo para inferencia.");
    }
    else
    {
        Print("Aviso: Modelo ONNX no encontrado o error de carga. El EA funcionará solo con RL y Genética.");
    }
    
    if (!InpTrainingMode)
    {
        G_SignalAgent.SetEpsilon(0.01);
        G_ManageAgent.SetEpsilon(0.01);
    }
        
    Print("QuantFlow V2 Ready. Hybrid AI Architecture Active.");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if (!G_RiskManager.IsTradingAllowed()) return;

    static datetime last_day = 0;
    datetime current_day = iTime(_Symbol, PERIOD_D1, 0);
    if (current_day != last_day) { G_RiskManager.OnNewDay(); last_day = current_day; }

    CheckActiveTrades();

    // Actualización del GA cada 4 horas
    static datetime last_ga_update = 0;
    if (TimeCurrent() - last_ga_update > 4 * 3600)
    {
        G_GAOptimizer.Evolve();
        last_ga_update = TimeCurrent();
    }

    if (!G_TradeActive)
    {
        // 1. Obtener estado del mercado
        vector current_state = CFeatureFactory::GetState(_Symbol, _Period);
        
        // 2. Consultar Agente RL (Signal)
        int sig_action = G_SignalAgent.GetAction(current_state);
        
        // 3. Consultar Modelo ONNX (Confirmación) - OPCIONAL pero RECOMENDADO
        if (G_LSTMModel.IsReady())
        {
             // Para simplificar, asumimos que FeatureFactory o una nueva función nos da los buffers
             // necesarios para el LSTM (Returns, RSI, ATR de las últimas 20 velas).
             // AQUÍ SIMULAMOS DATOS PARA QUE EL CÓDIGO COMPILE Y MUESTRE LA LÓGICA
             // En producción real, debes implementar CFeatureFactory::GetHistoryBuffers(...)
             
             vector v_rsi, v_atr, v_ret; // Vacíos por ahora
             // int lstm_pred = G_LSTMModel.Predict(v_rsi, v_atr, v_ret);
             
             // Lógica de Fusión de Decisiones:
             // if (lstm_pred != sig_action) return; // VETO si no coinciden
             // O simplemente usarlo como peso extra.
        }
        
        if (sig_action == 0) return;
        
        // 4. Consultar Agente RL (Risk Management)
        int man_action = G_ManageAgent.GetAction(current_state);
        
        // 5. Consultar Optimizador Genético (Parámetros Base)
        int opt_sl = 0, opt_tp = 0;
        G_GAOptimizer.GetBestParameters(opt_sl, opt_tp);
        
        double tp_mult = 1.0, sl_mult = 1.0;
        if(man_action == 1) { tp_mult = 1.3; sl_mult = 0.9; } // Agresivo (Mejor RR)
        if(man_action == 2) { tp_mult = 0.9; sl_mult = 1.1; } // Conservador (Más seguro)
        
        double final_sl = opt_sl * sl_mult;
        double final_tp = opt_tp * tp_mult;
        
        if (sig_action == 1) // Buy
        {
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - final_sl * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + final_tp * _Point;
            double lots = G_RiskManager.GetMaxLotSize(InpRiskPerTrade, final_sl);
            
            if (G_Trade.Buy(lots, _Symbol, 0, sl, tp, "QF_AI_BUY"))
            {
                G_LastState = current_state;
                G_LastSigAction = 1;
                G_LastManAction = man_action;
                G_TradeActive = true;
            }
        }
        else if (sig_action == 2) // Sell
        {
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_BID) + final_sl * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_BID) - final_tp * _Point;
            double lots = G_RiskManager.GetMaxLotSize(InpRiskPerTrade, final_sl);
            
            if (G_Trade.Sell(lots, _Symbol, 0, sl, tp, "QF_AI_SELL"))
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
            
            // Usamos Remember (Experience Replay) para aprendizaje estable DQN
            G_SignalAgent.Remember(G_LastState, G_LastSigAction, reward, next_state, true);
            G_ManageAgent.Remember(G_LastState, G_LastManAction, reward, next_state, true);
            
            // Print("Trade Closed. Reward: ", reward);
        }
        G_TradeActive = false;
    }
}

bool PositionSelectByMagic(long magic)
{
    for(int i=PositionsTotal()-1; i>=0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket)) return true;
    }
    return false;
}
