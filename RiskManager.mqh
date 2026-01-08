//+------------------------------------------------------------------+
//|                                                  RiskManager.mqh |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property strict

#include <Trade\Trade.mqh>

class CRiskManager
{
private:
    double      m_max_daily_dd_pct;
    double      m_max_total_dd_pct;
    double      m_daily_start_equity;
    CTrade      m_trade;

public:
    CRiskManager() : m_max_daily_dd_pct(4.5), m_max_total_dd_pct(9.0) {}
    
    void SetLimits(double daily_dd, double total_dd)
    {
        m_max_daily_dd_pct = daily_dd;
        m_max_total_dd_pct = total_dd;
    }

    void OnNewDay()
    {
        m_daily_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        Print("New Day Started. Daily Start Equity: ", m_daily_start_equity);
    }

    bool IsTradingAllowed()
    {
        double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        
        // Check Daily Drawdown
        double daily_dd = (m_daily_start_equity - current_equity) / m_daily_start_equity * 100.0;
        if (daily_dd >= m_max_daily_dd_pct)
        {
            Print("Daily Drawdown Limit Reached: ", daily_dd, "%");
            CloseAll();
            return false;
        }

        // Check Total Drawdown
        double total_dd = (balance - current_equity) / balance * 100.0;
        if (total_dd >= m_max_total_dd_pct)
        {
            Print("Total Drawdown Limit Reached: ", total_dd, "%");
            CloseAll();
            return false;
        }

        return true;
    }

    void CloseAll()
    {
        for (int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if (ticket > 0)
                m_trade.PositionClose(ticket);
        }
    }

    double GetMaxLotSize(double risk_pct, double stop_loss_points)
    {
        if (stop_loss_points <= 0) return 0.01;
        
        double margin_free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
        double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        
        if (tick_value == 0 || tick_size == 0) return 0.01;

        double risk_amount = AccountInfoDouble(ACCOUNT_BALANCE) * (risk_pct / 100.0);
        double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        
        double lot_size = risk_amount / (stop_loss_points * (tick_value / tick_size));
        
        // Normalize
        lot_size = MathFloor(lot_size / lot_step) * lot_step;
        
        double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
        
        if (lot_size < min_lot) lot_size = min_lot;
        if (lot_size > max_lot) lot_size = max_lot;
        
        return lot_size;
    }
};
