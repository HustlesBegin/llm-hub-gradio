# 🧠 LLM Hub - Gradio App

Aplicación en **Gradio** que permite seleccionar distintos **modelos de lenguaje**
(Groq, Google AI, OpenRouter, Ollama) para realizar tareas de **Chat**,
**Traducción**, **Resumen**, **Análisis de Sentimiento**, y **procesamiento por lotes de archivos `.txt` (Batch y Q&A)**.

---

## ⚙️ Funcionalidades principales

### 🗣️ Chat interactivo

Permite mantener una conversación libre con el modelo seleccionado.

* Puedes definir un **prompt de sistema** para controlar el tono o rol del asistente.
* Mantiene historial de conversación.
* Muestra **tiempo de inferencia** y **métricas** básicas de tokens.

### 🌍 Traducción automática

Traduce texto entre **Español ↔ Inglés** con precisión y naturalidad.

* Opción para elegir dirección de traducción.
* Ideal para textos académicos o explicativos.

### 🧾 Resumen de texto

Genera resúmenes automáticos con control de longitud (número de palabras).

* Resume artículos, párrafos o documentos largos.
* Muestra métricas de comparación entre texto de entrada y salida.

### ❤️ Análisis de Sentimiento

Clasifica textos como **Positivo**, **Negativo** o **Neutro**, con una breve justificación.

* Útil para reseñas, encuestas o feedback.
* Incluye métricas rápidas y tiempo de procesamiento.

### 📂 Batch TXT (procesamiento por lotes)

Permite subir uno o más archivos `.txt` para procesarlos automáticamente.

* Cada archivo se **resume individualmente**.
* Devuelve una tabla con resultados y métricas por documento.
* Ideal para resúmenes masivos o análisis de grandes conjuntos de texto.

### ❓ Q&A TXT (preguntas sobre archivos)

Haz preguntas específicas sobre el contenido de uno o varios archivos `.txt`.

* Implementa un método de **recuperación de contexto** (divide el texto en fragmentos y busca los más relevantes).
* Devuelve respuestas precisas para cada documento.
* Incluye **preguntas sugeridas** para orientación rápida.

---

## 🧩 Características adicionales

* **Interfaz adaptativa** con modo claro/oscuro automático.
* **Selección dinámica** de proveedor y modelo compatible.
* **Manejo de errores robusto** (timeouts, límites de uso, conexión, API keys inválidas, etc.).
* **Reintentos automáticos** en caso de error temporal (HTTP 429/503).
* **Métricas de rendimiento:**

  * Tiempo total de inferencia.
  * Palabras, caracteres y tokens estimados (entrada/salida).
* **Markdown enriquecido** con scroll para salidas extensas.

---

## 🚀 Ejecución local

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu-usuario/llm-hub-gradio.git
   cd llm-hub-gradio
   ```

2. Crea y activa un entorno virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Crea un archivo `.env` en la raíz con tus API keys:

   ```bash
   GROQ_API_KEY=tu_clave
   GEMINI_API_KEY=tu_clave
   OPEN_ROUTER_API_KEY=tu_clave
   ```

5. Ejecuta la aplicación:

   ```bash
   python app.py
   ```

   La interfaz se abrirá automáticamente en
   👉 [http://localhost:7860](http://localhost:7860)
