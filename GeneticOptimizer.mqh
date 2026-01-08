//+------------------------------------------------------------------+
//|                                             GeneticOptimizer.mqh |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property strict

struct SIndividual
{
   int sl;
   int tp;
   double fitness;
};

class CGeneticOptimizer
{
private:
   SIndividual m_population[];
   int         m_pop_size;
   int         m_generations;
   int         m_history_bars;
   
   int         m_min_sl, m_max_sl;
   int         m_min_tp, m_max_tp;
   
   SIndividual m_best_ever;

public:
   CGeneticOptimizer(int pop_size=30)
   {
      m_pop_size = pop_size;
      m_generations = 5;
      m_history_bars = 500;
      
      // Valores por defecto
      m_min_sl = 100; m_max_sl = 1000;
      m_min_tp = 100; m_max_tp = 2000;
      
      ArrayResize(m_population, m_pop_size);
      m_best_ever.sl = 300;
      m_best_ever.tp = 600;
      m_best_ever.fitness = -DBL_MAX;
   }

   // Configura los límites según los inputs del usuario
   void SetBounds(int min_sl, int max_sl, int min_tp, int max_tp)
   {
      m_min_sl = min_sl; m_max_sl = max_sl;
      m_min_tp = min_tp; m_max_tp = max_tp;
   }

   void Initialize()
   {
      for(int i=0; i<m_pop_size; i++)
      {
         m_population[i].sl = m_min_sl + MathRand() % (m_max_sl - m_min_sl + 1);
         m_population[i].tp = m_min_tp + MathRand() % (m_max_tp - m_min_tp + 1);
         m_population[i].fitness = 0;
      }
   }

   double Evaluate(SIndividual &ind)
   {
      double total_profit = 0;
      int trades = 0;
      
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(_Symbol, _Period, 1, m_history_bars, rates) < m_history_bars) return 0;

      for(int i=m_history_bars-2; i>=0; i -= 15) 
      {
         double entry_price = rates[i].close;
         double sl_p = ind.sl * _Point;
         double tp_p = ind.tp * _Point;
         
         // Simulación de Buy
         for(int j=i-1; j>=0; j--)
         {
            if(rates[j].low <= entry_price - sl_p) { total_profit -= ind.sl; break; }
            if(rates[j].high >= entry_price + tp_p) { total_profit += ind.tp; break; }
         }
         
         // Simulación de Sell
         for(int j=i-1; j>=0; j--)
         {
            if(rates[j].high >= entry_price + sl_p) { total_profit -= ind.sl; break; }
            if(rates[j].low <= entry_price - tp_p) { total_profit += ind.tp; break; }
         }
         trades += 2;
      }
      
      return (trades > 0) ? total_profit / trades : 0;
   }

   void Evolve()
   {
      for(int g=0; g<m_generations; g++)
      {
         for(int i=0; i<m_pop_size; i++)
         {
            m_population[i].fitness = Evaluate(m_population[i]);
            if(m_population[i].fitness > m_best_ever.fitness)
               m_best_ever = m_population[i];
         }
         
         SIndividual next_gen[];
         ArrayResize(next_gen, m_pop_size);
         
         for(int i=0; i<m_pop_size; i++)
         {
            int p1 = MathRand() % m_pop_size;
            int p2 = MathRand() % m_pop_size;
            SIndividual parent = (m_population[p1].fitness > m_population[p2].fitness) ? m_population[p1] : m_population[p2];
            
            next_gen[i] = parent;
            if(MathRand() % 100 < 30) // Mutación 30%
            {
               next_gen[i].sl += (MathRand() % 101) - 50;
               next_gen[i].tp += (MathRand() % 201) - 100;
               
               if(next_gen[i].sl < m_min_sl) next_gen[i].sl = m_min_sl;
               if(next_gen[i].sl > m_max_sl) next_gen[i].sl = m_max_sl;
               if(next_gen[i].tp < m_min_tp) next_gen[i].tp = m_min_tp;
               if(next_gen[i].tp > m_max_tp) next_gen[i].tp = m_max_tp;
            }
         }
         ArrayCopy(m_population, next_gen);
      }
   }

   void GetBestParameters(int &sl, int &tp)
   {
      sl = m_best_ever.sl;
      tp = m_best_ever.tp;
   }
};
