# app.py
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
import gradio.themes as themes

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
        "models": ["llama2"],
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # placeholder
    },
}

TASKS = ["Chat", "Traducción", "Resumen"]
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
        client = make_client(provider)

        if task == "Chat":
            msgs = []
            if sys_prompt.strip():
                msgs.append({"role": "system", "content": sys_prompt.strip()})
            else:
                msgs.append({"role": "system", "content": "Eres un asistente útil, claro y preciso."})
            msgs.extend(chat_history)
            msgs.append({"role": "user", "content": input_text})

            out = chat_completion(client, model, msgs, temperature)
            new_hist = chat_history + [{"role": "user", "content": input_text},
                                       {"role": "assistant", "content": out}]
            return out, new_hist

        elif task == "Traducción":
            msgs = build_translation_prompt(input_text, direction)
            out = chat_completion(client, model, msgs, temperature)
            return out, chat_history

        elif task == "Resumen":
            msgs = build_summary_prompt(input_text, lang="es", target_words=int(target_words))
            out = chat_completion(client, model, msgs, temperature)
            return out, chat_history

        return "Tarea no soportada.", chat_history
    except Exception as e:
        return f"Error: {e}", chat_history

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
    def on_provider_change(provider: str):
        models = MODELS_CONFIG[provider]["models"]
        return gr.update(choices=models, value=models[0] if models else None)

    dd_provider.change(on_provider_change, dd_provider, dd_model)

    def on_run(provider, model, task, input_text, direction, sys_prompt, temperature, target_words, hist):
        return run_task(provider, model, task, input_text, direction, sys_prompt, temperature, target_words, hist)

    btn_run.click(
        fn=on_run,
        inputs=[dd_provider, dd_model, rd_task, tb_in, rd_dir, tb_sys, sl_temp, sl_words, state_hist],
        outputs=[tb_out, state_hist],
    )

    btn_clear.click(lambda: [], None, state_hist)

    gr.Examples(
        examples=[
            [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Traducción",
             "La economía circular promueve el uso eficiente de recursos.", "ES → EN", "", 0.2, 120],
            [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Resumen",
             "La fotosíntesis convierte energía lumínica en química almacenada en glucosa...", "ES → EN", "", 0.2, 80],
            [default_provider, MODELS_CONFIG[default_provider]["models"][0], "Chat",
             "Explica la diferencia entre aprendizaje supervisado y no supervisado.", "ES → EN",
             "Eres un tutor experto en ML.", 0.2, 120],
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
