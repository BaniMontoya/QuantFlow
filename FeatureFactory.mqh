//+------------------------------------------------------------------+
//|                                               FeatureFactory.mqh |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property strict

class CFeatureFactory
{
public:
    static vector GetState(string symbol, ENUM_TIMEFRAMES tf)
    {
        vector state;
        state.Init(10); // 10 features
        
        // 1. RSI (Normalized 0..1)
        int rsi_handle = iRSI(symbol, tf, 14, PRICE_CLOSE);
        double rsi_val[1];
        CopyBuffer(rsi_handle, 0, 0, 1, rsi_val);
        state[0] = rsi_val[0] / 100.0;
        
        // 2. ATR (Normalized by price)
        int atr_handle = iATR(symbol, tf, 14);
        double atr_val[1];
        CopyBuffer(atr_handle, 0, 0, 1, atr_val);
        double price = SymbolInfoDouble(symbol, SYMBOL_BID);
        state[1] = (atr_val[0] / price) * 1000.0; // Scale up for visibility
        
        // 3. Distance from MA (Normalized)
        int ma_handle = iMA(symbol, tf, 50, 0, MODE_EMA, PRICE_CLOSE);
        double ma_val[1];
        CopyBuffer(ma_handle, 0, 0, 1, ma_val);
        state[2] = (price - ma_val[0]) / (atr_val[0] > 0 ? atr_val[0] : 1.0);
        
        // 4. Momentum (Close[0] - Close[5])
        double close_prices[6];
        CopyClose(symbol, tf, 0, 6, close_prices);
        state[3] = (close_prices[5] - close_prices[0]) / (atr_val[0] > 0 ? atr_val[0] : 1.0);
        
        // 5. Hour of day (Cyclical Sin)
        MqlDateTime dt;
        TimeCurrent(dt);
        state[4] = MathSin(2.0 * M_PI * dt.hour / 24.0);
        state[5] = MathCos(2.0 * M_PI * dt.hour / 24.0);
        
        // 6. ADX (Strength of trend)
        int adx_handle = iADX(symbol, tf, 14);
        double adx_buffer[1];
        CopyBuffer(adx_handle, 0, 0, 1, adx_buffer);
        state[6] = adx_buffer[0] / 100.0;
        
        // 7. Equity DD (from Risk Manager perspective)
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        state[7] = (balance - equity) / balance;
        
        // 8. Consecutive wins/losses (approx)
        // (Just a placeholder for now)
        state[8] = 0.5; 
        
        // 9. Standard Deviation
        int std_handle = iStdDev(symbol, tf, 20, 0, MODE_SMA, PRICE_CLOSE);
        double std_val[1];
        CopyBuffer(std_handle, 0, 0, 1, std_val);
        state[9] = std_val[0] / (atr_val[0] > 0 ? atr_val[0] : 1.0);

        return state;
    }
};
