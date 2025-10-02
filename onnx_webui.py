# onnx_webui.py
import gradio as gr
import subprocess
import threading
import queue
import os
import signal
from pathlib import Path
from gradio import Timer

DIFF_ROOT = Path("/workspace/DiffSinger")
ONNX_ENV  = Path("/workspace/venv_onnx")      # ← 你的已有环境路径
REQ_FILE  = DIFF_ROOT / "requirements-onnx.txt"

# ---------- 环境工具 ----------
def get_python() -> str:
    """返回环境内 python 可执行文件；不存在则空"""
    exe = ONNX_ENV / "bin" / "python"        # Linux
    if not exe.exists():
        exe = ONNX_ENV / "Scripts" / "python.exe"  # Windows
    return str(exe) if exe.exists() else ""

def is_env_ready() -> bool:
    return get_python() != ""

# ---------- 实时日志器 ----------
class LiveLog:
    def __init__(self):
        self.proc = None; self.q = queue.Queue(); self.hist = []
    def start(self, cmd: list, cwd: Path):
        if self.proc and self.proc.poll() is None: return
        self.hist.clear()
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._reader, daemon=True).start()
    def _reader(self):
        for line in iter(self.proc.stdout.readline, ""): self.q.put(line)
        self.proc.wait()
    def stop(self):
        if self.proc and self.proc.poll() is None: self.proc.terminate(); self.proc.wait()
        self.proc = None
    def read(self) -> str:
        new_lines = []
        while True:
            try: new_lines.append(self.q.get_nowait())
            except queue.Empty: break
        self.hist.extend(new_lines)
        return "".join(self.hist)

live_log = LiveLog()

# ---------- ONNX 导出 ----------
def export_onnx(mode: str, exp: str, ckpt: str, out: str, freeze: str, export_spk: list):
    python = get_python()
    if not python:
        return " 未找到 ONNX 环境，请先点击「进入环境」"
    cmd = [python, "scripts/export.py", mode,
           "--exp", exp,
           "--out", out or f"artifacts/{exp}"]
    if ckpt:
        cmd += ["--ckpt", ckpt]
    if freeze:
        cmd += [freeze]
    for spk in export_spk:
        cmd += ["--export_spk", spk]
    live_log.start(cmd, DIFF_ROOT)
    return f" {mode} ONNX 导出已启动..."

def enter_env():
    """仅检测，不创建"""
    if is_env_ready():
        return " 已进入 ONNX 环境"
    return " 环境不存在，请确认 /workspace/venv_onnx 已创建"

def exit_env():
    """仅提示，无实际退出（conda 无法 exit）"""
    return " 已退出 ONNX 环境（conda deactivate）"

# ---------- Gradio 页面 ----------
def build_onnx_tab():
    with gr.Blocks(title="ONNX 导出中心") as blk:
        gr.Markdown("##  DiffSinger ONNX 导出中心（进入已有环境）")

        # 环境状态灯
        with gr.Row():
            enter_btn = gr.Button(" 进入环境", variant="secondary")
            exit_btn  = gr.Button(" 退出环境", variant="stop")
            status_tb = gr.Textbox(value=enter_env(), label="环境状态", interactive=False, lines=1)
            enter_btn.click(fn=enter_env, outputs=status_tb)
            exit_btn.click(fn=exit_env, outputs=status_tb)

        # 通用日志框
        log = gr.Textbox(label="实时日志", lines=18, max_lines=20, interactive=False)

        # 声学导出
        with gr.Group():
            gr.Markdown("###  声学模型 ONNX 导出")
            with gr.Row():
                acoustic_exp = gr.Textbox(label="实验名", placeholder="my_acoustic_exp")
                acoustic_ckpt = gr.Textbox(label="ckpt步数（空=自动最大）", placeholder="")
                acoustic_out = gr.Textbox(label="输出目录（空=artifacts/实验名）", placeholder="")
                acoustic_freeze = gr.Checkbox(value=False, label="冻结 velocity（不推荐）")
            acoustic_spk = gr.Textbox(value="base", label="导出说话人（逗号分隔）", placeholder="base,bohe")
            acoustic_btn = gr.Button("导出声学ONNX", variant="primary")
            acoustic_btn.click(
                lambda exp, ckpt, out, freeze, spk: export_onnx(
                    "acoustic", exp, ckpt, out, "--freeze_velocity" if freeze else "", spk.split(",")),
                inputs=[acoustic_exp, acoustic_ckpt, acoustic_out, acoustic_freeze, acoustic_spk],
                outputs=log, queue=True
            )

        # 唱法导出
        with gr.Group():
            gr.Markdown("###  唱法模型 ONNX 导出")
            with gr.Row():
                variance_exp = gr.Textbox(label="实验名", placeholder="my_variance_exp")
                variance_ckpt = gr.Textbox(label="ckpt步数（空=自动最大）", placeholder="")
                variance_out = gr.Textbox(label="输出目录（空=artifacts/实验名）", placeholder="")
                variance_freeze_glide = gr.Checkbox(value=False, label="冻结 glide（OpenUtau兼容）")
            variance_spk = gr.Textbox(value="base", label="导出说话人（逗号分隔）", placeholder="base,bohe")
            variance_btn = gr.Button("导出唱法ONNX", variant="primary")
            variance_btn.click(
                lambda exp, ckpt, out, fg, spk: export_onnx(
                    "variance", exp, ckpt, out, "--freeze_glide" if fg else "", spk.split(",")),
                inputs=[variance_exp, variance_ckpt, variance_out, variance_freeze_glide, variance_spk],
                outputs=log, queue=True
            )

        # 停止 & 刷新
        stop_btn = gr.Button(" 停止当前任务", variant="stop")
        stop_btn.click(fn=live_log.stop, outputs=None)

        timer = Timer(0.5, active=True)
        timer.tick(fn=live_log.read, outputs=log, queue=True)

    return blk