# app.py
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
import gradio.themes as themes
import time, re, json
import math, pathlib

load_dotenv()

# ==========
# Credenciales desde entorno
# ==========
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# ==========
# Config central (como pediste)
# ==========
MODELS_CONFIG = {
    "Groq": {
        "models": ["llama-3.1-8b-instant",
                   "openai/gpt-oss-20b"],
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
    },
    "Google AI": {
        "models": ["gemini-2.0-flash", 
                   "gemini-2.5-flash"],
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
    },
    "OpenRouter": {
        "models": ["alibaba/tongyi-deepresearch-30b-a3b:free", 
                   "deepseek/deepseek-chat-v3.1:free", 
                   "qwen/qwen3-coder:free"],
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPEN_ROUTER_API_KEY,
    },
    "Ollama (Local)": {
        "models": ["llama2", "gemma2:9b"],
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # placeholder
    },
}

TASKS = ["Chat", "Traducción", "Resumen", "Sentimiento", "Batch TXT", "Q&A TXT"]
TRANSLATION_DIRECTIONS = ["ES → EN", "EN → ES"]

# ==========
# Utilidades
# ==========
def make_client(provider: str) -> OpenAI:
    conf = MODELS_CONFIG[provider]
    base_url = conf["base_url"]
    api_key = conf["api_key"]
    if not api_key and provider != "Ollama (Local)":
        raise RuntimeError(f"Falta API key para {provider}. Configura la variable de entorno.")
    return OpenAI(base_url=base_url, api_key=api_key or "ollama")

def chat_completion(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

def chat_completion_with_retry(client: OpenAI, model: str, messages: List[Dict[str, str]],
                               temperature: float = 0.2,
                               max_retries: int = 2, backoff: float = 1.8):
    """
    Llama al modelo con pequeños reintentos para errores transitorios:
    - 429 (rate limit)
    - timeouts / errores de conexión
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            msg = str(e).lower()
            transient = (
                "429" in msg or
                "rate limit" in msg or
                "temporarily rate-limited" in msg or
                "timed out" in msg or
                "timeout" in msg or
                "connection" in msg or
                "temporarily unavailable" in msg or
                "service unavailable" in msg or
                "502" in msg or "503" in msg or "504" in msg
            )
            if attempt < max_retries and transient:
                # backoff exponencial suave: 0s, 1.8s, 3.24s...
                time.sleep(backoff ** attempt if attempt > 0 else 0.0)
                last_exc = e
                continue
            last_exc = e
            break
    raise last_exc

def explain_exception(e: Exception, provider: str, model: str) -> str:
    text = str(e)
    low = text.lower()

    # Intenta extraer código de estado si viene en el string
    code = None
    m = re.search(r"error code:\s*(\d+)", text, flags=re.I)
    if m:
        code = int(m.group(1))

    def is_status(n): return code == n or str(n) in low

    # Vacío se maneja antes, aquí filtramos resto:
    if is_status(401) or "unauthorized" in low or "invalid_api_key" in low:
        return "🔑 **Autenticación fallida (401/403)**: revisa tu API key y permisos."

    if is_status(403):
        return "🛑 **Prohibido (403)**: tu clave no tiene permisos para este recurso."

    if is_status(404) or "model_not_found" in low:
        return f"🔎 **Modelo no encontrado (404)**: `{model}` no está disponible en {provider}."

    if is_status(408) or "timeout" in low or "timed out" in low:
        return "⏳ **Timeout**: el proveedor tardó demasiado. Reintenta, cambia de modelo o baja la carga."

    if is_status(429) or "rate limit" in low or "temporarily rate-limited" in low:
        hint = ""
        base = MODELS_CONFIG.get(provider, {}).get("base_url", "").lower()
        if "openrouter.ai" in base:
            hint = (
                "\n\n💡 Sugerencias (OpenRouter `:free`):\n"
                "- Espera 10–30 s y vuelve a intentar.\n"
                "- Cambia a otro modelo `:free` o proveedor.\n"
                "- Agrega tu propia API key en OpenRouter para cuota propia."
            )
        return f"⛔ **Límite de tasa (429)**: `{model}` está temporalmente saturado.{hint}"

    if is_status(502) or is_status(503) or is_status(504) or "service unavailable" in low:
        return "🌩️ **Proveedor en mantenimiento/sobrecarga (5xx)**: intenta de nuevo en breve o cambia de modelo."

    if "connection" in low or "failed to establish a new connection" in low:
        return "🌐 **Error de conexión**: verifica tu internet o la `base_url` del proveedor."

    # Fallback
    return f"⚠️ Error inesperado: {text}"

def build_translation_prompt(text: str, direction: str) -> List[Dict[str, str]]:
    if direction == "ES → EN":
        sys = "You are a professional translator. Translate fluent and faithful Spanish → English."
        user = f"Traduce al inglés con naturalidad y precisión:\n\n{text}"
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    else:
        sys = "Eres un traductor profesional. Traduce de inglés a español con naturalidad y precisión."
        user = f"Translate to Spanish, preserving tone and meaning:\n\n{text}"
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def build_summary_prompt(text: str, lang: str = "es", target_words: int = 120) -> List[Dict[str, str]]:
    if lang.lower().startswith("en"):
        sys = "You are a world-class summarizer. Be concise and faithful."
        user = f"Summarize in ~{target_words} words, preserving key facts, in clear English:\n\n{text}"
    else:
        sys = "Eres un sintetizador de primer nivel. Sé conciso y fiel."
        user = f"Resume en ~{target_words} palabras, preservando ideas clave, en español claro:\n\n{text}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def build_sentiment_prompt(text: str, lang: str = "es") -> List[Dict[str, str]]:
    if lang.startswith("en"):
        sys = "You are a precise sentiment analyst. Return only one word: Positive, Neutral, or Negative."
        user = f"Text:\n{text}\n\nRespond with: Positive, Neutral, or Negative. Optionally add a short reason."
    else:
        sys = "Eres un analista de sentimiento. Devuelve solo una palabra: Positivo, Neutro o Negativo."
        user = f"Texto:\n{text}\n\nResponde con: Positivo, Neutro o Negativo. Opcional: una breve razón."
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def count_words(s: str) -> int:
    return len(s.split())

def estimate_tokens_simple(s: str) -> int:
    # Estimación rápida y agnóstica de tokens (~4 chars/token)
    return max(1, math.ceil(len(s) / 4))

def format_metrics(title: str, input_text: str, output_text: str, elapsed: float) -> str:
    in_chars  = len(input_text)
    out_chars = len(output_text)
    in_words  = count_words(input_text)
    out_words = count_words(output_text)
    in_tok    = estimate_tokens_simple(input_text)
    out_tok   = estimate_tokens_simple(output_text)
    return (
        f"\n\n---\n"
        f"### 📊 {title}\n"
        f"- ⏱️ **Tiempo de inferencia:** {elapsed:.2f} s\n"
        f"- 📝 **Input:** {in_words} palabras · {in_chars} chars · ~{in_tok} tokens\n"
        f"- 🧾 **Output:** {out_words} palabras · {out_chars} chars · ~{out_tok} tokens\n"
    )

def run_batch_txt(provider, model, temperature, sys_prompt, direction, target_words, files) -> str:
    if not files:
        return "⚠️ No se subieron archivos .txt."

    client = make_client(provider)
    rows = []
    for f in files:
        # Validación de extensión (cuando hay nombre original)
        fname = getattr(f, "orig_name", None) or getattr(f, "name", None) or getattr(f, "path", None) or "archivo.txt"
        if not str(fname).lower().endswith(".txt"):
            rows.append((pathlib.Path(str(fname)).name, "❌ No es .txt", "-"))
            continue

        text, read_err = read_txt_file(f)
        if read_err:
            rows.append((pathlib.Path(str(fname)).name, f"❌ {read_err}", "-"))
            continue

        # Por defecto: resumir cada archivo (puedes cambiar a traducción/sentimiento)
        msgs = build_summary_prompt(text, lang="es", target_words=int(target_words))
        t0 = time.time()
        try:
            resp = chat_completion_with_retry(make_client(provider), model, msgs, temperature)
            out = resp.choices[0].message.content
            elapsed = time.time() - t0
            metrics = format_metrics("Métricas", text, out, elapsed)
            rows.append((pathlib.Path(str(fname)).name, out, metrics))
        except Exception as e:
            rows.append((pathlib.Path(str(fname)).name, explain_exception(e, provider, model), "-"))


    # Render como Markdown (tabla simple)
    md = ["## 📂 Resultados Batch TXT\n", "| Archivo | Resultado |", "|---|---|"]
    for name, result, m in rows:
        # Compactar cada resultado; evita pipes rompiendo tablas
        safe = result.replace("\n", "<br>")
        md.append(f"| `{name}` | {safe} |")
        if m != "-":
            md.append(f"\n{m}\n")
    return "\n".join(md)

def read_txt_file(f) -> Tuple[str, str]:
    """
    Devuelve (texto, error). Si error != "", falló la lectura/decodificación.
    Soporta objetos UploadedFile de Gradio (con .path/.name) o rutas/strings.
    """
    try:
        # 1) Ruta por atributo .path
        if hasattr(f, "path") and f.path:
            p = f.path
        # 2) Ruta por atributo .name
        elif hasattr(f, "name") and isinstance(f.name, str):
            p = f.name
        # 3) Directamente una cadena de ruta
        elif isinstance(f, str):
            p = f
        else:
            return "", "Formato de archivo no reconocido (sin .path/.name)."

        # Lee bytes y decodifica con fallback
        with open(p, "rb") as fh:
            data = fh.read()
        try:
            return data.decode("utf-8"), ""
        except UnicodeDecodeError:
            return data.decode("latin-1"), ""

    except Exception as e:
        return "", f"Error leyendo archivo: {e}"

def build_qa_prompt(context: str, question: str, lang: str = "es") -> List[Dict[str, str]]:
    if lang.startswith("en"):
        sys = ("You answer questions using ONLY the provided context. "
               "If the answer is not present, say 'I can't find it in the provided text.'")
        user = (f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer in English, concise and faithful to the context.")
    else:
        sys = ("Respondes preguntas usando SOLO el contexto proporcionado. "
               "Si la respuesta no está, di: 'No lo encuentro en el texto proporcionado'.")
        user = (f"Contexto:\n{context}\n\n"
                f"Pregunta: {question}\n"
                f"Responde en español, de forma concisa y fiel al contexto.")
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def run_batch_qa_txt(provider, model, temperature, question, files, 
                     chunk_size=1500, overlap=200, top_k=3, max_total_chars=9000) -> str:
    """
    Q&A por archivo: divide en trozos, selecciona top-K por similitud con la pregunta
    y construye el contexto solo con esos trozos para cada archivo.
    """
    if not files:
        return "⚠️ No se subieron archivos .txt."
    if not question or not question.strip():
        return "⚠️ Escribe una pregunta para Q&A."

    rows = []
    for f in files:
        fname = getattr(f, "orig_name", None) or getattr(f, "name", None) or getattr(f, "path", None) or "archivo.txt"
        if not str(fname).lower().endswith(".txt"):
            rows.append((pathlib.Path(str(fname)).name, "❌ No es .txt", "-"))
            continue

        text, read_err = read_txt_file(f)
        if read_err:
            rows.append((pathlib.Path(str(fname)).name, f"❌ {read_err}", "-"))
            continue

        # 1) Trocear
        chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
        # 2) Puntuar por similitud simple
        scored = sorted(((simple_score(question, c), idx, c) for idx, c in enumerate(chunks)),
                        key=lambda x: x[0], reverse=True)
        # 3) Elegir top-K no vacíos
        top = [c for s, _, c in scored if s > 0][:top_k]
        if not top:
            # si no hay solapamiento, toma el primer trozo (fallback)
            top = chunks[:1]

        # 4) Limitar tamaño total del contexto para no exceder tokens
        context = ""
        for c in top:
            if len(context) + len(c) > max_total_chars:
                break
            context += c + "\n\n"

        # 5) Prompt de Q&A
        msgs = build_qa_prompt(context, question, lang="es")

        t0 = time.time()
        try:
            resp = chat_completion_with_retry(make_client(provider), model, msgs, temperature)
            answer = resp.choices[0].message.content
            elapsed = time.time() - t0
            metrics = format_metrics("Métricas", question, answer, elapsed)
            rows.append((pathlib.Path(str(fname)).name, answer, metrics))
        except Exception as e:
            rows.append((pathlib.Path(str(fname)).name, explain_exception(e, provider, model), "-"))

    # Render en Markdown
    md = ["## ❓ Q&A sobre archivos TXT\n", "| Archivo | Respuesta |", "|---|---|"]
    for name, result, m in rows:
        safe = result.replace("\n", "<br>")
        md.append(f"| `{name}` | {safe} |")
        if m != "-":
            md.append(f"\n{m}\n")
    return "\n".join(md)


def split_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 200):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        if j == n: break
        i = j - overlap  # solapamiento
        if i < 0: i = 0
    return chunks

_word_re = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def simple_score(query: str, chunk: str) -> int:
    # Scoring naive por solapamiento de palabras (rápido y sin dependencias)
    q_words = set(w.lower() for w in _word_re.findall(query))
    c_words = set(w.lower() for w in _word_re.findall(chunk))
    return len(q_words & c_words)

def run_task(
    provider: str,
    model: str,
    task: str,
    input_text: str,
    direction: str,
    sys_prompt: str,
    temperature: float,
    target_words: int,
    chat_history: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]]]:
    try:
        # Validación de entrada vacía
        if not input_text.strip():
            return "⚠️ Error: el texto de entrada está vacío.", chat_history

        client = make_client(provider)
        t0 = time.time()

        if task == "Chat":
            msgs = []
            if sys_prompt.strip():
                msgs.append({"role": "system", "content": sys_prompt.strip()})
            else:
                msgs.append({"role": "system", "content": "Eres un asistente útil, claro y preciso."})
            msgs.extend(chat_history)
            msgs.append({"role": "user", "content": input_text})

            resp = chat_completion_with_retry(client, model, msgs, temperature)
            out = resp.choices[0].message.content
            new_hist = chat_history + [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": out},
            ]

        elif task == "Traducción":
            msgs = build_translation_prompt(input_text, direction)
            resp = chat_completion_with_retry(client, model, msgs, temperature)
            out = resp.choices[0].message.content
            new_hist = chat_history

        elif task == "Resumen":
            msgs = build_summary_prompt(input_text, lang="es", target_words=int(target_words))
            resp = chat_completion_with_retry(client, model, msgs, temperature)
            out = resp.choices[0].message.content
            new_hist = chat_history

        elif task == "Sentimiento":
            msgs = build_sentiment_prompt(input_text, lang="es")
            t0 = time.time()
            resp = chat_completion_with_retry(client, model, msgs, temperature)
            out = resp.choices[0].message.content
            elapsed = time.time() - t0
            out += format_metrics("Métricas", input_text, out, elapsed)
            new_hist = chat_history

        else:
            return "⚠️ Tarea no soportada.", chat_history

        elapsed = time.time() - t0
        out += f"\n\n---\n⏱️ **Tiempo de inferencia:** {elapsed:.2f} s"
        return out, new_hist

    except Exception as e:
        # Mensaje claro según el tipo de error
        friendly = explain_exception(e, provider, model)
        return friendly, chat_history

# ==========
# UI Gradio
# ==========
all_providers = list(MODELS_CONFIG.keys())
default_provider = all_providers[0]
default_models = MODELS_CONFIG[default_provider]["models"]

with gr.Blocks(
    title="LLM Hub: Chat · Traducción · Resumen",
    theme=themes.Default(primary_hue="blue", neutral_hue="slate")
) as demo:
    
    gr.Markdown("# 🧠 LLM Hub\nSelecciona **Proveedor → Modelo → Tarea**. Ingresa tu texto y obtén el resultado.")

    with gr.Row():
        dd_provider = gr.Dropdown(
            label="Proveedor", 
            choices=all_providers, 
            value=default_provider,
            allow_custom_value=False,
            filterable=False,
            )
        dd_model = gr.Dropdown(
            label="Modelo", 
            choices=default_models, 
            value=default_models[0],
            allow_custom_value=False,
            filterable=False,
            )
        
        rd_task = gr.Radio(
            label="Tarea", 
            choices=TASKS, 
            value="Traducción")
        
    files_box = gr.Files(
                    label="Archivos .txt (procesamiento por lotes)",
                    file_types=[".txt"],
                    visible=False
                )
    help_batch = gr.Markdown(value="> Sube uno o más .txt y usa la tarea **Batch TXT**.", visible=False)

    qa_question = gr.Textbox(
                        label="Pregunta para Q&A sobre archivo(s)",
                        placeholder="Escribe tu pregunta sobre el/los archivo(s) .txt…",
                        visible=False,
                        lines=2
                    )
    suggested_q = gr.Radio(
        label="Preguntas sugeridas",
        choices=[
            "¿Cuál es el objetivo principal que menciona el documento?",
            "¿Qué política ambiental se propone?",
            "¿Qué diferencias se plantean entre los paradigmas?"
        ],
        visible=False
    )

    with gr.Accordion("Opciones avanzadas", open=False):
        tb_sys = gr.Textbox(label="System prompt (solo Chat)", placeholder="Define rol/tono...", lines=2)
        sl_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=0.2)
        sl_words = gr.Slider(label="Longitud objetivo resumen (palabras)", minimum=30, maximum=300, step=10, value=120)

    rd_dir = gr.Radio(label="Dirección de traducción", choices=TRANSLATION_DIRECTIONS, value="ES → EN")

    with gr.Row():
        tb_in = gr.Textbox(
            label="Input", 
            lines=10, 
            placeholder="Escribe tu texto aquí…"
        )

        tb_out = gr.Markdown(
            value="> 💬 *El resultado aparecerá aquí...*",
            label="Output",
            elem_id="markdown-output"
        )

    with gr.Row():
        btn_run = gr.Button("▶ Ejecutar", variant="primary")
        btn_clear = gr.Button("🧹 Limpiar historial (Chat)")

    state_hist = gr.State([])  # historial tipo List[Dict[str,str]]

    # Eventos
    def on_pick_suggested(q):
        # Copia la sugerida al textbox de pregunta
        return gr.update(value=q)

    suggested_q.change(on_pick_suggested, suggested_q, qa_question)

    def on_provider_change(provider: str):
        models = MODELS_CONFIG[provider]["models"]
        return gr.update(choices=models, value=models[0] if models else None)

    dd_provider.change(on_provider_change, dd_provider, dd_model)

    def on_run(provider, model, task, input_text, direction, sys_prompt, temperature, target_words, hist, files, question):
        if task == "Batch TXT":
            out = run_batch_txt(provider, model, temperature, sys_prompt, direction, target_words, files)
            return out, hist
        elif task == "Q&A TXT":
            out = run_batch_qa_txt(provider, model, temperature, question, files)
            return out, hist
        else:
            return run_task(provider, model, task, input_text, direction, sys_prompt, temperature, target_words, hist)

    btn_run.click(
        fn=on_run,
        inputs=[dd_provider, dd_model, rd_task, tb_in, rd_dir, tb_sys, sl_temp, sl_words, state_hist, files_box, qa_question],
        outputs=[tb_out, state_hist],
    )


    btn_clear.click(lambda: [], None, state_hist)

    def on_task_change(task):
        show_batch = (task == "Batch TXT")
        show_qa = (task == "Q&A TXT")
        return [
            gr.update(visible=show_batch),  # files_box
            gr.update(visible=show_batch),  # help_batch
            gr.update(visible=show_qa),     # qa_question
            gr.update(visible=show_qa),     # suggested_q
        ]

    rd_task.change(on_task_change, rd_task, [files_box, help_batch, qa_question, suggested_q])



    gr.Examples(
        examples=[
            [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Traducción",
            "La economía circular promueve el uso eficiente de recursos.", "ES → EN", "", 0.2, 120],
            [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Resumen",
            "La fotosíntesis convierte energía lumínica en química almacenada en glucosa...", "ES → EN", "", 0.2, 80],
            [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Chat",
            "Explica la diferencia entre aprendizaje supervisado y no supervisado.", "ES → EN",
            "Eres un tutor experto en ML.", 0.2, 120],
            # Ejemplos nuevos opcionales:
            # [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Sentimiento",
            #  "Este producto es fantástico, superó mis expectativas.", "ES → EN", "", 0.2, 120],
            # [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Batch TXT",
            #  "", "ES → EN", "", 0.2, 120],
        ],
        inputs=[dd_provider, dd_model, rd_task, tb_in, rd_dir, tb_sys, sl_temp, sl_words],
        outputs=[tb_out],
        label="Ejemplos rápidos",
    )


    gr.Markdown(
        """
**Notas rápidas**
- **Groq / Google AI / OpenRouter**: asegúrate de definir las variables de entorno correspondientes.
- **Ollama**: ejecuta `ollama serve` y ten descargado el modelo (`ollama pull llama2`).
- El **historial** solo aplica para *Chat*. Traducción y Resumen son *single-turn*.
        """
    )

# Estilos personalizados
demo.css = """
#markdown-output {
    height: 400px;
    overflow-y: auto;
    border-radius: 6px;
    padding: 10px;
}

/* Estilos modo oscuro */
:root[data-theme=dark] #markdown-output {
    border: 1px solid #444;
    background-color: #1a1a1a;
    color: #eee;
}

/* Estilos modo claro */
:root[data-theme=light] #markdown-output {
    border: 1px solid #ccc;
    background-color: #fafafa;
    color: #111;
}
"""

if __name__ == "__main__":
    demo.launch()
