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
        state[8] = 0.5; 
        
        // 9. Standard Deviation
        int std_handle = iStdDev(symbol, tf, 20, 0, MODE_SMA, PRICE_CLOSE);
        double std_val[1];
        CopyBuffer(std_handle, 0, 0, 1, std_val);
        state[9] = std_val[0] / (atr_val[0] > 0 ? atr_val[0] : 1.0);

        return state;
    }
    
    // Obtiene los vectores históricos necesarios para el modelo ONNX (LSTM)
    static bool GetHistoryVectors(string symbol, ENUM_TIMEFRAMES tf, int lookback, vector &r_rsi, vector &r_atr, vector &r_ret)
    {
        // Redimensionar vectores
        if(!r_rsi.Resize(lookback) || !r_atr.Resize(lookback) || !r_ret.Resize(lookback)) return false;
        
        // Obtener Handles
        int h_rsi = iRSI(symbol, tf, 14, PRICE_CLOSE);
        int h_atr = iATR(symbol, tf, 14);
        
        double buf_rsi[], buf_atr[], buf_close[];
        
        // Nota: CopyBuffer devuelve [0] como la más reciente si NO está como serie, o depende de configuración.
        // Pero para vectores MQL5, el índice 0 suele ser el inicio.
        // Vamos a extraer (Lookback + 1) para calcular retorno.
        
        if(CopyBuffer(h_rsi, 0, 1, lookback, buf_rsi) < lookback) return false;
        if(CopyBuffer(h_atr, 0, 1, lookback, buf_atr) < lookback) return false;
        if(CopyClose(symbol, tf, 1, lookback + 1, buf_close) < lookback + 1) return false;
        
        // Asignar a vectores
        // Orden: El modelo Python se entrenó con secuencias cronológicas (Antiguo -> Nuevo)
        // CopyBuffer defecto: [0] es Oldest si usamos orden normal.
        // Comprobemos: Si pedimos start=1 count=20, nos da de ayer a hoy.
        
        for(int i=0; i<lookback; i++)
        {
            r_rsi[i] = buf_rsi[i];
            r_atr[i] = buf_atr[i];
            
            // Retorno: (Precio[i] - Precio[i-1]) / Precio[i-1]
            // buf_close tiene 21 elementos. buf_close[0] es el más viejo.
            // retorno[0] corresponde al tiempo de r_rsi[0].
            double prev_close = buf_close[i];
            double curr_close = buf_close[i+1];
            
            if(prev_close != 0)
                r_ret[i] = (curr_close - prev_close) / prev_close;
            else
                r_ret[i] = 0.0;
        }
        
        return true;
    }
};
