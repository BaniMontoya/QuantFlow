//+------------------------------------------------------------------+
//|                                                     RL_Agent.mqh |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property strict

// Arquitectura de Red Neuronal para Aprendizaje Continuo en MQL5
class CRLAgent
{
private:
    matrix      m_weights_h;    // (hidden_size x input_size)
    matrix      m_weights_o;    // (output_size x hidden_size)
    vector      m_bias_h;
    vector      m_bias_o;
    
    double      m_learning_rate;
    double      m_gamma;        
    double      m_epsilon;      
    
    int         m_input_size;
    int         m_hidden_size;
    int         m_output_size;
    string      m_name;

public:
    CRLAgent(string name, int input_size, int hidden_size, int output_size)
    {
        m_name = name;
        m_input_size = input_size;
        m_hidden_size = hidden_size;
        m_output_size = output_size;
        
        m_learning_rate = 0.02;
        m_gamma = 0.9;
        m_epsilon = 0.1;
        
        m_weights_h.Init(hidden_size, input_size);
        m_weights_o.Init(output_size, hidden_size);
        
        double std_h = MathSqrt(2.0/input_size);
        double std_o = MathSqrt(2.0/hidden_size);

        for(ulong i=0; i<m_weights_h.Rows(); i++)
            for(ulong j=0; j<m_weights_h.Cols(); j++)
                m_weights_h[i][j] = ((double)MathRand()/32767.0 - 0.5) * 2.0 * std_h;

        for(ulong i=0; i<m_weights_o.Rows(); i++)
            for(ulong j=0; j<m_weights_o.Cols(); j++)
                m_weights_o[i][j] = ((double)MathRand()/32767.0 - 0.5) * 2.0 * std_o;
                
        m_bias_h.Init(hidden_size);
        m_bias_o.Init(output_size);
        m_bias_h.Fill(0.0);
        m_bias_o.Fill(0.0);
    }

    // Predicción Forward usando MatMul explícito para evitar errores de tipo
    vector Predict(vector &state)
    {
        // matrix.MatMul(vector) retorna un vector
        vector h = m_weights_h.MatMul(state);
        h = h + m_bias_h;
        h.Activation(h, AF_RELU);
        
        vector o = m_weights_o.MatMul(h);
        o = o + m_bias_o;
        return o;
    }

    int GetAction(vector &state)
    {
        if (((double)MathRand()/32767.0) < m_epsilon)
            return MathRand() % m_output_size;
            
        vector q_values = Predict(state);
        return (int)q_values.ArgMax();
    }

    // Entrenamiento Online
    void Train(vector &state, int action, double reward, vector &next_state, bool done)
    {
        vector current_q = Predict(state);
        vector next_q = Predict(next_state);
        
        double target = reward;
        if (!done)
            target += m_gamma * next_q.Max();
            
        double error = target - current_q[action];
        
        vector h = m_weights_h.MatMul(state);
        h = h + m_bias_h;
        h.Activation(h, AF_RELU);
        
        // 1. Actualizar Capa de Salida
        for(int i=0; i<m_hidden_size; i++)
        {
            m_weights_o[action][i] += m_learning_rate * error * h[i];
        }
        m_bias_o[action] += m_learning_rate * error;
        
        // 2. Actualizar Capa Oculta (Backprop manual simplificado)
        for(int i=0; i<m_hidden_size; i++)
        {
            if(h[i] > 0)
            {
                double grad = error * m_weights_o[action][i];
                for(int j=0; j<m_input_size; j++)
                {
                    m_weights_h[i][j] += m_learning_rate * grad * state[j];
                }
                m_bias_h[i] += m_learning_rate * grad;
            }
        }
    }
    
    void SetEpsilon(double eps) { m_epsilon = eps; }
    void SetLearningRate(double lr) { m_learning_rate = lr; }
};
