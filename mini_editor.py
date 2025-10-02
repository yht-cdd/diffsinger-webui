# mini_editor.py
import gradio as gr
import yaml
from pathlib import Path

CFG_ROOT = Path(__file__).resolve().parent / "DiffSinger" / "configs"

def list_yaml():
    return [str(p.relative_to(CFG_ROOT)) for p in CFG_ROOT.rglob("*.yaml")]

def load_yaml(rel_path: str):
    path = CFG_ROOT / rel_path
    return path.read_text(encoding="utf-8"), str(path)

def save_yaml(rel_path: str, text: str):
    path = CFG_ROOT / rel_path
    try:
        yaml.safe_load(text)   # 简单语法校验
    except Exception as e:
        return f" YAML 语法错误：{e}"
    path.write_text(text, encoding="utf-8")
    return f" 已覆盖保存：{path}"

def build_mini_editor():
    with gr.Blocks(title="配置文本编辑器") as editor:
        gr.Markdown("###  选择配置 → 直接编辑 → 保存覆盖")
        with gr.Row():
            file_dd = gr.Dropdown(choices=list_yaml(), label="配置文件", scale=4)
            reload_btn = gr.Button(" 刷新列表", scale=1)
        editor = gr.Code(language="yaml", label="YAML 内容", lines=30)
        save_btn = gr.Button(" 保存并覆盖原文件", variant="primary")
        log = gr.Textbox(label="日志", interactive=False)

        file_dd.change(fn=lambda rel: (load_yaml(rel)[0], rel), inputs=file_dd, outputs=[editor, file_dd])
        reload_btn.click(fn=lambda: gr.Dropdown(choices=list_yaml()), outputs=file_dd)
        save_btn.click(fn=save_yaml, inputs=[file_dd, editor], outputs=log)

    return editor