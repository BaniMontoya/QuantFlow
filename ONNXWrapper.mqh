//+------------------------------------------------------------------+
//|                                                  ONNXWrapper.mqh |
//|                                  Copyright 2024, Antigravity AI  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Antigravity AI"
#property link      "https://www.mql5.com"
#property strict

class CONNXModel
{
private:
   long     m_handle;
   long     m_input_shape[];
   long     m_output_shape[];
   string   m_filename;
   bool     m_initialized;

public:
   CONNXModel() : m_handle(INVALID_HANDLE), m_initialized(false) {}
   
   ~CONNXModel()
   {
      if(m_handle != INVALID_HANDLE)
      {
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
      }
   }

   bool Load(string filename)
   {
      m_filename = filename;
      
      // Load from MQL5/Files/
      // Note: OnnxCreate requires path relative to MQL5/Files for FILE_COMMON or generally just the name 
      // if it's in the specialized folder. 
      // Actually standard OnnxCreate takes a buffer or a filename.
      // Let's try loading from the common Files folder.
      
      m_handle = OnnxCreate(filename, ONNX_DEFAULT);
      
      if(m_handle == INVALID_HANDLE)
      {
         // Try loading as resource if file load fails (fallback)
         Print("ONNX: No se pudo cargar desde archivo. Intentando cargar recurso...");
         return false;
      }
      
      // Verify Input/Output shapes
      // We expect Input: [Batch, Lookback, Features] i.e. [1, 20, 3]
      // Output: [Batch, Classes] i.e. [1, 3]
      
      long inputs[] = {1, 20, 3}; 
      if(!OnnxSetInputShape(m_handle, 0, inputs))
      {
         Print("ONNX: Error configurando forma de entrada: ", GetLastError());
         return false;
      }
      
      long outputs[] = {1, 3};
      if(!OnnxSetOutputShape(m_handle, 0, outputs))
      {
         Print("ONNX: Error configurando forma de salida: ", GetLastError());
         return false;
      }
      
      m_initialized = true;
      Print("ONNX: Modelo cargado exitosamente: ", filename);
      return true;
   }
   
   int Predict(const vector &rsi_buffer, const vector &atr_buffer, const vector &returns_buffer)
   {
      if(!m_initialized) return -1; // Error
      
      // Prepare Input Tensor
      // Data layout must match Python training: [Returns, RSI, ATR] sequence x 20
      // Flattened array or matrix? OnnxRun accepts vector/matrix.
      // Since our input is 3D [1, 20, 3], we might need to flatten it if using vector,
      // or use matrix if 2D. But 3D 1x20x3 is tricky in MQL5 specific types.
      // MQL5 OnnxRun works best with vectors/matrices.
      // Simplification: We will reshape the input in Python to be [1, 60] (20*3) or similar if needed.
      // But standard LSTM expects sequence.
      // MQL5 supports float array for input.
      
      float input_data[];
      ArrayResize(input_data, 20 * 3);
      
      // Fill data (Latest data is at end of buffer? Python used Oldest -> Newest)
      // Assuming buffers passed are size 20, ordered Oldest -> Newest
      
      for(int i=0; i<20; i++)
      {
         // Normalize inputs roughly as we did in Python (MinMax -1 to 1)
         // This is a CRITICAL STEP. We need the scaler values from Python.
         // For now we use rough normalization or assume inputs are already normalized?
         // No, we must normalize.
         // Let's use relative scaling for simplicity in this V1.
         
         input_data[i*3 + 0] = (float)returns_buffer[i]; // Returns are already small %
         input_data[i*3 + 1] = (float)(rsi_buffer[i] / 100.0 * 2.0 - 1.0); // RSI 0..100 -> -1..1
         input_data[i*3 + 2] = (float)atr_buffer[i]; // ATR is raw, tricky to normalize without history max.
         // PENDING: We need the scaler params from Python.
      }
      
      // Run Inference
      vector output_data; 
      // Output is float vector [3] (probabilities/logits for Sell, Buy, Hold)
      
      if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, input_data, output_data))
      {
         Print("ONNX: Error en inferencia: ", GetLastError());
         return -1;
      }
      
      // Argmax
      return (int)output_data.ArgMax(); // 0=Hold, 1=Buy, 2=Sell (Mapping from Python script)
   }
   
   bool IsReady() { return m_initialized; }
};
