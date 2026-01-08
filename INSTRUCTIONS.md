# QuantFlow - AI Expert Advisor

Este repositorio contiene un sistema de trading híbrido que combina **Reinforcement Learning**, **Algoritmos Genéticos** y **Deep Learning (LSTM)**.

## Estructura del Proyecto
- `QuantFlow.mq5`: EA principal.
- `RL_Agent.mqh`: Agente de Reinforcement Learning (DQN con Experience Replay).
- `GeneticOptimizer.mqh`: Optimizador genético interno para SL/TP dinámicos.
- `ONNXWrapper.mqh`: Wrapper para cargar modelos LSTM (.onnx).
- `train_lstm.py`: Script de Python para entrenar el modelo Maestro.
- `QuantFlow_TrainingData.csv`: Datos históricos para entrenamiento.

## Guía de Configuración en el Servidor

### 1. Preparar el entorno de Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy torch scikit-learn onnx onnxruntime
```

### 2. Entrenar el Modelo LSTM
Ejecuta el script de entrenamiento para procesar los datos históricos y generar el cerebro ONNX:
```bash
python train_lstm.py
```
Esto generará un archivo llamado `QuantFlow_LSTM.onnx`.

### 3. Instalación en MetaTrader 5
1. Copia el archivo generado `QuantFlow_LSTM.onnx` a la carpeta `/MQL5/Files/` de tu terminal MT5.
2. Compila `QuantFlow.mq5` en el MetaEditor.
3. Arrastra el EA a un gráfico (preferiblemente XAUUSD M1).

## Notas Técnicas
- El modelo LSTM espera una secuencia de 20 velas.
- El EA se auto-optimiza genéticamente cada 4 horas.
- El aprendizaje por refuerzo ocurre en tiempo real después de cada operación cerrada.
