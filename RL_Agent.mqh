//+------------------------------------------------------------------+
//|                                                     RL_Agent.mqh |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property strict

// Arquitectura de Red Neuronal con Experience Replay para Aprendizaje Robusto
struct SExperience
{
    vector state;
    int    action;
    double reward;
    vector next_state;
    bool   done;
};

class CRLAgent
{
private:
    matrix      m_weights_h;
    matrix      m_weights_o;
    vector      m_bias_h;
    vector      m_bias_o;
    
    double      m_learning_rate;
    double      m_gamma;        
    double      m_epsilon;      
    
    int         m_input_size;
    int         m_hidden_size;
    int         m_output_size;
    
    // Experience Replay Buffer
    SExperience m_memory[];
    int         m_mem_ptr;
    int         m_mem_size;
    int         m_batch_size;

public:
    CRLAgent(string name, int input_size, int hidden_size, int output_size)
    {
        m_input_size = input_size;
        m_hidden_size = hidden_size;
        m_output_size = output_size;
        
        m_learning_rate = 0.01;
        m_gamma = 0.95;
        m_epsilon = 0.1;
        
        m_mem_size = 500;   // Guarda las últimas 500 experiencias
        m_batch_size = 32;  // Entrena en lotes de 32
        m_mem_ptr = 0;
        ArrayResize(m_memory, m_mem_size);
        
        m_weights_h.Init(hidden_size, input_size);
        m_weights_o.Init(output_size, hidden_size);
        
        // Xavier/Glorot Initialization
        double std_h = MathSqrt(2.0/(input_size + hidden_size));
        double std_o = MathSqrt(2.0/(hidden_size + output_size));

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

    vector Predict(vector &state)
    {
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

    // Guardar experiencia en memoria
    void Remember(vector &state, int action, double reward, vector &next_state, bool done)
    {
        m_memory[m_mem_ptr].state = state;
        m_memory[m_mem_ptr].action = action;
        m_memory[m_mem_ptr].reward = reward;
        m_memory[m_mem_ptr].next_state = next_state;
        m_memory[m_mem_ptr].done = done;
        
        m_mem_ptr = (m_mem_ptr + 1) % m_mem_size;
        
        // Cada vez que guardamos, entrenamos un pequeño lote (Replay)
        ExperienceReplay();
    }

    void ExperienceReplay()
    {
        for(int b=0; b<m_batch_size; b++)
        {
            int idx = MathRand() % m_mem_size;
            if(m_memory[idx].state.Size() == 0) continue; 
            
            SExperience exp = m_memory[idx];
            
            vector current_q = Predict(exp.state);
            vector next_q = Predict(exp.next_state);
            
            double target = exp.reward;
            if (!exp.done)
                target += m_gamma * next_q.Max();
                
            double error = target - current_q[exp.action];
            
            vector h = m_weights_h.MatMul(exp.state);
            h = h + m_bias_h;
            h.Activation(h, AF_RELU);
            
            for(int i=0; i<m_hidden_size; i++)
                m_weights_o[exp.action][i] += m_learning_rate * error * h[i];
            m_bias_o[exp.action] += m_learning_rate * error;
            
            if(MathAbs(error) > 0.0001)
            {
                for(int i=0; i<m_hidden_size; i++)
                {
                    if(h[i] > 0)
                    {
                        double grad = error * m_weights_o[exp.action][i];
                        for(int j=0; j<m_input_size; j++)
                            m_weights_h[i][j] += m_learning_rate * grad * exp.state[j];
                        m_bias_h[i] += m_learning_rate * grad;
                    }
                }
            }
        }
    }
    
    void SetEpsilon(double eps) { m_epsilon = eps; }
};
