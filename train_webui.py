#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiffSinger 训练模块 WebUI（逐行累计版）
"""
import os
import gradio as gr
import subprocess
import threading
import queue
from pathlib import Path
from gradio import Timer

WORKSPACE = Path("/workspace/DiffSinger")
os.chdir(WORKSPACE)

# ---------- 实时日志器 ----------
class LiveLog:
    def __init__(self):
        self.proc  = None
        self.q     = queue.Queue()
        self.hist  = []          # 历史缓存

    def start(self, cmd: list, cwd: Path):
        if self.proc and self.proc.poll() is None:
            return
        self.hist.clear()        # 新任务清空历史
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in iter(self.proc.stdout.readline, ""):
            self.q.put(line)
        self.proc.wait()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
        self.proc = None

    def read(self) -> str:
        """返回 历史 + 新行，实现逐行累计"""
        new_lines = []
        while True:
            try:
                new_lines.append(self.q.get_nowait())
            except queue.Empty:
                break
        self.hist.extend(new_lines)
        return "".join(self.hist)   # ← 关键：每次都返回全文

# ---------- 全局实例 ----------
live_log = LiveLog()

# ---------- 流式命令 ----------
def run_live(cmd: list, log: gr.Textbox):
    live_log.stop()
    live_log.start(cmd, WORKSPACE)
    return " 任务已启动，日志实时刷新中..."

# ---------- 页面 ----------
def build_train_tab():
    with gr.Blocks(title="DiffSinger Training WebUI") as blk:
        gr.Markdown("#  DiffSinger 训练")

        log = gr.Textbox(label="日志", lines=18, max_lines=20, interactive=False)

        # 声学预处理
        with gr.Group():
            gr.Markdown("##  声学预处理")
            acoustic_config = gr.Textbox(label="声学配置文件路径", placeholder="configs/config_acoustic.yaml")
            acoustic_pre_btn = gr.Button(" 运行声学预处理", variant="primary")
            acoustic_pre_btn.click(
                lambda config: run_live(["python", "scripts/binarize.py", "--config", config], log) if config else " 路径为空",
                inputs=acoustic_config, outputs=log
            )

        # 声学训练
        with gr.Group():
            gr.Markdown("##  声学模型训练")
            acoustic_config2 = gr.Textbox(label="声学配置文件路径", placeholder="configs/config_acoustic.yaml")
            acoustic_exp = gr.Textbox(label="实验名称", placeholder="my_acoustic_exp")
            acoustic_reset = gr.Checkbox(value=True, label="重置训练（--reset）")
            acoustic_train_btn = gr.Button(" 开始声学训练", variant="primary")
            acoustic_train_btn.click(
                lambda cfg, exp, reset: run_live(
                    ["python", "scripts/train.py", "--config", cfg, "--exp_name", exp] + (["--reset"] if reset else []), log),
                inputs=[acoustic_config2, acoustic_exp, acoustic_reset], outputs=log
            )

        # 唱法预处理
        with gr.Group():
            gr.Markdown("##  唱法预处理")
            variance_config = gr.Textbox(label="唱法配置文件路径", placeholder="configs/config_variance.yaml")
            variance_pre_btn = gr.Button(" 运行唱法预处理", variant="primary")
            variance_pre_btn.click(
                lambda config: run_live(["python", "scripts/binarize.py", "--config", config], log) if config else " 路径为空",
                inputs=variance_config, outputs=log
            )

        # 唱法训练
        with gr.Group():
            gr.Markdown("##  唱法模型训练")
            variance_config2 = gr.Textbox(label="唱法配置文件路径", placeholder="configs/config_variance.yaml")
            variance_exp = gr.Textbox(label="实验名称", placeholder="my_variance_exp")
            variance_reset = gr.Checkbox(value=True, label="重置训练（--reset）")
            variance_train_btn = gr.Button(" 开始唱法训练", variant="primary")
            variance_train_btn.click(
                lambda cfg, exp, reset: run_live(
                    ["python", "scripts/train.py", "--config", cfg, "--exp_name", exp] + (["--reset"] if reset else []), log),
                inputs=[variance_config2, variance_exp, variance_reset], outputs=log
            )

        # 停止 & 刷新
        stop_btn = gr.Button(" 停止当前任务", variant="stop")
        stop_btn.click(fn=live_log.stop, outputs=None)

        timer = Timer(0.5, active=True)
        timer.tick(fn=live_log.read, outputs=log, queue=True)

    return blk

# ---------- 单独运行 ----------
if __name__ == "__main__":
    print("→ 启动 DiffSinger 训练 WebUI（逐行累计），0.0.0.0:7862")
    demo = gr.Blocks(title="DiffSinger Training LiveLog")
    with demo:
        build_train_tab()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=True)