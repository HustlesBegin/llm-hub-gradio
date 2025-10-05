# 🧠 LLM Hub - Gradio App

Aplicación en **Gradio** que permite seleccionar distintos modelos de lenguaje 
(Groq, Google AI, OpenRouter, Ollama) para realizar tareas de **Chat**, 
**Traducción** y **Resumen**.

## 🚀 Ejecución local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/llm-hub-gradio.git
   cd llm-hub-gradio
2. Crea un entorno virtual y activa:
    python -m venv venv
    source venv/bin/activate  # en Windows: venv\Scripts\activate
3. Instala dependencias:
    pip install -r requirements.txt
4. Crea un archivo .env en la raíz con tus API keys:
    GROQ_API_KEY=tu_clave
    GEMINI_API_KEY=tu_clave
    OPEN_ROUTER_API_KEY=tu_clave
5. Ejecuta la app:
    python app.py
    La interfaz se abrirá en http://localhost:7860
