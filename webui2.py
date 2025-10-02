#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gradio as gr
from pathlib import Path
import pandas as pd
# ---------- 工具 ----------
from pathlib import Path
from mini_editor import build_mini_editor
from train_webui import build_train_tab
from onnx_webui import build_onnx_tab
#实时输出
# webui2.py 顶部（全局）
import threading, queue, subprocess
from gradio import Timer

import threading, queue, subprocess

import subprocess
#执行命令
def run_cmd(cmd: str, timeout: int = 60) -> str:
    """
    执行命令并返回输出
    :param cmd: 要执行的命令
    :param timeout: 超时时间（秒）
    :return: 命令输出或错误信息
    """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"   错误：{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"   命令超时：{cmd}"
    except Exception as e:
        return f"   执行失败：{str(e)}"

#全局路径查找
import subprocess
from pathlib import Path

def find_and_run_script(script_name: str, base_path: Path, args: str, timeout: int = 60, max_depth: int = 3) -> str:
    """
    递归在基础路径及其子目录中查找脚本并运行。
    :param script_name: 脚本文件名（如 "build_dataset.py"）
    :param base_path: 基础路径（Path对象）
    :param args: 命令行参数（如 "--dir /path"）
    :param timeout: 超时时间（秒）
    :param max_depth: 最大搜索深度（默认3级）
    :return: 执行日志或错误信息
    """
    def search_script(directory: Path, current_depth: int) -> Path:
        """递归搜索脚本文件"""
        if current_depth > max_depth:
            return None
        # 检查当前目录
        script_path = directory / script_name
        if script_path.exists():
            return script_path
        # 递归搜索子目录
        for subdir in directory.iterdir():
            if subdir.is_dir():
                found = search_script(subdir, current_depth + 1)
                if found:
                    return found
        return None

    # 开始搜索
    found_path = search_script(base_path, 0)
    if found_path:
        script_dir = found_path.parent
        return run_cmd(f'cd "{script_dir}" && python {script_name} {args}', timeout)
    else:
        return f"   脚本不存在：{script_name} 在 {base_path} 及其子目录中（最大深度 {max_depth}）"
def mds_script(script_name: str, args: str, timeout: int = 60) -> str:
    """
    运行 MakeDiffSinger 目录下的脚本。
    :param script_name: 脚本文件名（如 "validate_lengths.py"）
    :param args: 命令行参数（如 "--dir /path"）
    :param timeout: 超时时间（秒）
    :return: 执行日志或错误信息
    """
    script_path = MDS_BASE / script_name
    if not script_path.exists():
        return f"   脚本不存在：{script_path}"
    cmd = f'cd "{MDS_BASE}" && python {script_name} {args}'
    return run_cmd(cmd, timeout)
class LiveRunner:
    """逐行实时日志器（Gradio 5.x 专用）"""
    def __init__(self):
        self.proc   = None
        self.q      = queue.Queue()
        self._hist  = []          # 累计历史

    def start(self, cmd: list, cwd: Path = Path(".")):
        if self.proc and self.proc.poll() is None:
            return                 # 已有任务
        self._hist.clear()
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
        """返回历史 + 新行（逐行累计）"""
        new_lines = []
        while True:
            try:
                new_lines.append(self.q.get_nowait())
            except queue.Empty:
                break
        self._hist.extend(new_lines)
        return "".join(self._hist)


class LiveLog:
    def __init__(self):
        self.proc  = None
        self.q     = queue.Queue()
        self._lock = threading.Lock()
        self._hist = []          # ← 新增：历史缓存

    def start(self, cmd: list, cwd: Path = Path(".")):
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return
        self._hist.clear()       # ← 启动时清空历史
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
        with self._lock:
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
        self._hist.extend(new_lines)          # ← 追加到历史
        return "".join(self._hist)            # ← 返回累计全文
# 全局实例（所有模块导入即可用）
live_log = LiveLog()

# 调用一次即可
live_runner = LiveRunner()   # 全局实例

def run_live(cmd: str, log: gr.Textbox):
    """启动任务 + 返回初始提示"""
    live_runner.stop()
    live_runner.start(cmd.split(), cwd=WORKSPACE)   # 必须用 list 形式
    return " 任务已启动，日志实时刷新中..."
# ============== 补充：config_tab 完整实现 ==============
def build_acoustic_subtab(): 
    import yaml
    from pathlib import Path

    TEMPLATE_FILE = Path(__file__).resolve().parent / "DiffSinger" / "configs" / "templates" / "config_acoustic.yaml"
    EXPORT_DIR    = TEMPLATE_FILE.parent

    # ----------- 工具：读/写 YAML -----------
    def load_template():
        if not TEMPLATE_FILE.exists():
            return None, f" 模板不存在：{TEMPLATE_FILE}"
        with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg, f" 已载入模板：{TEMPLATE_FILE}"

    def export_new_config(
        model_name, datasets_df, lr, bs, max_step,
        pe, pe_ckpt, hnsep_ckpt, vocoder_ckpt,
        use_eng, use_bre, use_voi, use_ten,
        use_ks, use_sp, use_lang, use_spk,
        num_lang, num_spk,
        key_shift_range, vel_range,
        out_name,
    ):
        try:
            cfg = {
                "base_config": "configs/acoustic.yaml",
                "dictionaries": {"zh": "dictionaries/opencpop-extension.txt"},
                "extra_phonemes": [],
                "merged_phoneme_groups": [],
                "datasets": [],
                "binary_data_dir": f"data/{model_name}/binary",
                "binarization_args": {"num_workers": 0},
                "pe": pe,
                "pe_ckpt": pe_ckpt if pe == "rmvpe" else "",
                "hnsep": "vr",
                "hnsep_ckpt": hnsep_ckpt,
                "vocoder": "NsfHifiGAN",
                "vocoder_ckpt": vocoder_ckpt,
                "use_lang_id": use_lang,
                "num_lang": int(num_lang),
                "use_spk_id": use_spk,
                "num_spk": int(num_spk),
                "use_energy_embed": use_eng,
                "use_breathiness_embed": use_bre,
                "use_voicing_embed": use_voi,
                "use_tension_embed": use_ten,
                "use_key_shift_embed": use_ks,
                "use_speed_embed": use_sp,
                "augmentation_args": {
                    "random_pitch_shifting": {"enabled": True, "range": key_shift_range, "scale": 0.75},
                    "fixed_pitch_shifting": {"enabled": False, "targets": [-5., 5.], "scale": 0.5},
                    "random_time_stretching": {"enabled": True, "range": vel_range, "scale": 0.75},
                },
                "diffusion_type": "reflow",
                "enc_ffn_kernel_size": 3,
                "use_rope": True,
                "use_shallow_diffusion": True,
                "T_start": 0.4,
                "T_start_infer": 0.4,
                "K_step": 300,
                "K_step_infer": 300,
                "backbone_type": "lynxnet",
                "backbone_args": {
                    "num_channels": 1024,
                    "num_layers": 6,
                    "kernel_size": 31,
                    "dropout_rate": 0.0,
                    "strong_cond": True,
                },
                "shallow_diffusion_args": {
                    "train_aux_decoder": True,
                    "train_diffusion": True,
                    "val_gt_start": False,
                    "aux_decoder_arch": "convnext",
                    "aux_decoder_args": {
                        "num_channels": 512,
                        "num_layers": 6,
                        "kernel_size": 7,
                        "dropout_rate": 0.1,
                    },
                    "aux_decoder_grad": 0.1,
                },
                "lambda_aux_mel_loss": 0.2,
                "optimizer_args": {"lr": float(lr)},
                "lr_scheduler_args": {
                    "scheduler_cls": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 10000,
                    "gamma": 0.75,
                },
                "max_batch_frames": 50000,
                "max_batch_size": int(bs),
                "max_updates": int(max_step),
                "num_valid_plots": 10,
                "val_with_vocoder": True,
                "val_check_interval": 2000,
                "num_ckpt_keep": 5,
                "permanent_ckpt_start": 120000,
                "permanent_ckpt_interval": 20000,
                "pl_trainer_devices": "auto",
                "pl_trainer_precision": "16-mixed",
            }

            for row in datasets_df.itertuples(index=False):
                cfg["datasets"].append({
                    "raw_data_dir": row.raw_data_dir,
                    "speaker": row.speaker,
                    "spk_id": int(row.spk_id),
                    "language": row.language,
                    "test_prefixes": [p.strip() for p in row.test_prefixes.split(",") if p.strip()],
                })

            out_file = EXPORT_DIR / f"new_{out_name}.yaml"
            with open(out_file, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            return f"  已导出：{out_file}"
        except Exception as e:
            return f"  导出失败：{str(e)}"

    # ---------------- 界面 ----------------
    with gr.Blocks(title="声学模型配置") as blk:
        gr.Markdown("## 🎛️ 声学模型配置生成器（自动检测模板）")

        # 顶部状态
        with gr.Row():
            load_st = gr.Textbox(value="", label="模板状态", interactive=False, scale=4)
            reload_btn = gr.Button("🔄 重新检测", scale=1)

        # 左侧：基础信息
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 基础信息")
                model_name = gr.Textbox(value="my_acoustic", label="模型名（new_xxx.yaml 前缀）", placeholder="my_acoustic")
                base_cfg   = gr.Textbox(value=str(TEMPLATE_FILE), interactive=False, label="模板路径")
                bin_dir    = gr.Textbox(value="", interactive=False, label="预处理输出目录")

            # 右侧：可调参数
            with gr.Column(scale=2):
                gr.Markdown("#### 可调训练参数")
                with gr.Row():
                    lr = gr.Number(value=0.0006, label="初始学习率")
                    bs = gr.Number(value=64, label="max_batch_size")
                    max_step = gr.Number(value=160000, label="max_updates")
                with gr.Row():
                    pe = gr.Radio(["parselmouth", "rmvpe"], value="parselmouth", label="音高提取器")
                    pe_ckpt = gr.Textbox(value="checkpoints/rmvpe/model.pt", label="rmvpe 模型路径")
                with gr.Row():
                    hnsep_ckpt = gr.Textbox(value="checkpoints/vr/model.pt", label="vr 模型路径")
                    vocoder_ckpt = gr.Textbox(value="checkpoints/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt", label="声码器 ckpt")
                with gr.Row():
                    use_eng = gr.Checkbox(value=False, label="use_energy_embed")
                    use_bre = gr.Checkbox(value=False, label="use_breathiness_embed")
                    use_voi = gr.Checkbox(value=False, label="use_voicing_embed")
                    use_ten = gr.Checkbox(value=False, label="use_tension_embed")
                with gr.Row():
                    use_ks = gr.Checkbox(value=True, label="use_key_shift_embed")
                    use_sp = gr.Checkbox(value=True, label="use_speed_embed")
                    use_lang = gr.Checkbox(value=False, label="多语言")
                    use_spk  = gr.Checkbox(value=False, label="多说话人")
                with gr.Row():
                    num_lang = gr.Number(value=1, precision=0, label="语言总数")
                    num_spk  = gr.Number(value=1, precision=0, label="说话人总数")
                with gr.Row():
                    key_shift_range = gr.Slider([-12, 12], value=(-5, 5), step=0.5, label="随机变调范围")
                    vel_range       = gr.Slider([0.1, 3], value=(0.5, 2), step=0.1, label="随机变速范围")

        # 数据集动态表格
        gr.Markdown("#### 数据集列表（动态增减）")
        with gr.Row():
            add_btn = gr.Button(" 添加行", scale=1)
            del_btn = gr.Button(" 删除末行", scale=1)
        ds_df = gr.Dataframe(
    headers=["raw_data_dir", "speaker", "spk_id", "language", "test_prefixes"],
    datatype=["str", "str", "number", "str", "str"],
    value=[["data/xxx1/raw", "speaker1", 0, "zh", "wav1,wav2"]],
    row_count=(1, 20),    # 最小 1 行，最大 20 行
    interactive=True,
)




        add_btn.click(
    lambda df: pd.concat([df, pd.DataFrame([["data/xxx/raw", "speaker", 0, "zh", "test"]], columns=df.columns)], ignore_index=True),
    inputs=ds_df,
    outputs=ds_df
)



        # 导出
        with gr.Row():
            out_name = gr.Textbox(value="my_acoustic", label="导出文件名前缀（不含 .yaml）")
            export_btn = gr.Button("📥 导出 new_xxx.yaml", variant="primary")
            export_log = gr.Textbox(label="日志", interactive=False)

        # 事件
        export_btn.click(
            export_new_config,
            inputs=[
                out_name, ds_df, lr, bs, max_step,
                pe, pe_ckpt, hnsep_ckpt, vocoder_ckpt,
                use_eng, use_bre, use_voi, use_ten,
                use_ks, use_sp, use_lang, use_spk,
                num_lang, num_spk,
                key_shift_range, vel_range,
                out_name,
            ],
            outputs=export_log,
        )

        # 启动时自动检测模板
        def on_load():
            cfg, msg = load_template()
            if cfg is None:
                return msg, "", ""
            return (
                msg,
                cfg.get("binary_data_dir", "").replace("/binary", "")[5:],  # 剥掉 data/xxx/binary → xxx
                cfg.get("optimizer_args", {}).get("lr", 0.0006),
            )
        blk.load(on_load, outputs=[load_st, model_name, lr])

    return blk
# ============== END config_tab ===============

def build_variance_subtab():
    import yaml, gradio as gr
    from pathlib import Path

    TEMPLATE_FILE = Path(__file__).resolve().parent / "DiffSinger" / "configs" / "templates" / "config_variance.yaml"
    EXPORT_DIR    = TEMPLATE_FILE.parent

    # ---------------- 工具 ----------------
    def load_var_template():
        if not TEMPLATE_FILE.exists():
            return None, f"  唱法模板不存在：{TEMPLATE_FILE}"
        with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg, f"  已载入唱法模板：{TEMPLATE_FILE}"

    def export_variance_config(
        model_name, datasets_df,
        lr, bs, max_step,
        pe, pe_ckpt, hnsep_ckpt,
        pred_dur, pred_pitch, pred_eng, pred_bre, pred_voi, pred_ten,
        eng_min, eng_max, bre_min, bre_max, voi_min, voi_max, ten_min, ten_max,
        use_melody, melody_hidden,
        use_glide, glide_types, glide_scale,
        out_prefix,
        dictionaries_config,    # 字典配置{语言: 路径}
        extra_phonemes_list,    # 特殊音素列表
        merged_phonemes_groups, # 合并音素组
        vocoder_type,          # 声码器类型
        vocoder_ckpt,          # 声码器检查点
        use_lang,              # 使用多语言
        num_lang,              # 语言数量
        use_spk,               # 使用多说话人
        num_spk,               # 说话人数量
    ):
        try:
            cfg = {
                "base_config": "configs/variance.yaml",
                "dictionaries": {"zh": "dictionaries/opencpop-extension.txt"},
                "extra_phonemes": [],
                "merged_phoneme_groups": [],
                "datasets": [],
                "binary_data_dir": f"data/{model_name}/binary",
                "binarization_args": {"num_workers": 0},
                "pe": pe,
                "pe_ckpt": pe_ckpt if pe == "rmvpe" else "",
                "hnsep": "vr",
                "hnsep_ckpt": hnsep_ckpt,
                "use_lang_id": False,
                "num_lang": 1,
                "use_spk_id": False,
                "num_spk": 1,
                "predict_dur": pred_dur,
                "predict_pitch": pred_pitch,
                "predict_energy": pred_eng,
                "predict_breathiness": pred_bre,
                "predict_voicing": pred_voi,
                "predict_tension": pred_ten,
                "energy_db_min": eng_min,
                "energy_db_max": eng_max,
                "breathiness_db_min": bre_min,
                "breathiness_db_max": bre_max,
                "voicing_db_min": voi_min,
                "voicing_db_max": voi_max,
                "tension_logit_min": ten_min,
                "tension_logit_max": ten_max,
                "enc_ffn_kernel_size": 3,
                "use_rope": True,
                "hidden_size": 256,
                "dur_prediction_args": {
                    "arch": "fs2",
                    "hidden_size": 512,
                    "dropout": 0.1,
                    "num_layers": 5,
                    "kernel_size": 3,
                    "log_offset": 1.0,
                    "loss_type": "mse",
                    "lambda_pdur_loss": 0.3,
                    "lambda_wdur_loss": 1.0,
                    "lambda_sdur_loss": 3.0,
                },
                "use_melody_encoder": use_melody,
                "melody_encoder_args": {
                    "hidden_size": melody_hidden,
                    "enc_layers": 4,
                },
                "use_glide_embed": use_glide,
                "glide_types": glide_types,
                "glide_embed_scale": glide_scale,
                "diffusion_type": "reflow",
                "pitch_prediction_args": {
                    "pitd_norm_min": -8.0,
                    "pitd_norm_max": 8.0,
                    "pitd_clip_min": -12.0,
                    "pitd_clip_max": 12.0,
                    "repeat_bins": 64,
                    "backbone_type": "wavenet",
                    "backbone_args": {
                        "num_layers": 20,
                        "num_channels": 256,
                        "dilation_cycle_length": 5,
                    },
                },
                "variances_prediction_args": {
                    "total_repeat_bins": 48,
                    "backbone_type": "wavenet",
                    "backbone_args": {
                        "num_layers": 10,
                        "num_channels": 192,
                        "dilation_cycle_length": 4,
                    },
                },
                "lambda_dur_loss": 1.0,
                "lambda_pitch_loss": 1.0,
                "lambda_var_loss": 1.0,
                "optimizer_args": {"lr": float(lr)},
                "lr_scheduler_args": {
                    "scheduler_cls": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 10000,
                    "gamma": 0.75,
                },
                "max_batch_frames": 80000,
                "max_batch_size": int(bs),
                "max_updates": int(max_step),
                "num_valid_plots": 10,
                "val_check_interval": 2000,
                "num_ckpt_keep": 5,
                "permanent_ckpt_start": 80000,
                "permanent_ckpt_interval": 10000,
                "pl_trainer_devices": "auto",
                "pl_trainer_precision": "16-mixed",
                "dictionaries": dictionaries_config,
                "extra_phonemes": extra_phonemes_list,
                "merged_phoneme_groups": merged_phonemes_groups,
                "vocoder": vocoder_type,
                "vocoder_ckpt": vocoder_ckpt,
                "use_lang_id": use_lang,
                "num_lang": num_lang,
                "use_spk_id": use_spk,
                "num_spk": num_spk,
            }

            for row in datasets_df.itertuples(index=False):
                cfg["datasets"].append({
                    "raw_data_dir": row.raw_data_dir,
                    "speaker": row.speaker,
                    "spk_id": int(row.spk_id),
                    "language": row.language,
                    "test_prefixes": [p.strip() for p in row.test_prefixes.split(",") if p.strip()],
                })

            out_file = EXPORT_DIR / f"new_{out_prefix}_variance.yaml"
            with open(out_file, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            return f"  已导出唱法配置：{out_file}"
        except Exception as e:
            return f"  导出失败：{str(e)}"

    # ---------------- 界面：唱法配置副标签页 ----------------
    with gr.Blocks() as blk:
        gr.Markdown("###  唱法模型参数（variance）")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 基础信息")
                model_name = gr.Textbox(value="my_var", label="模型名（new_xxx_variance.yaml 前缀）")
                load_st = gr.Textbox(value="", label="模板状态", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("#### 可调训练参数")
                lr = gr.Number(value=0.0006, label="初始学习率")
                bs = gr.Number(value=48, label="max_batch_size")
                max_step = gr.Number(value=160000, label="max_updates")

        with gr.Row():
            pe = gr.Radio(["parselmouth", "rmvpe"], value="parselmouth", label="音高提取器")
            pe_ckpt = gr.Textbox(value="checkpoints/rmvpe/model.pt", label="rmvpe 模型路径")
            hnsep_ckpt = gr.Textbox(value="checkpoints/vr/model.pt", label="vr 模型路径")

        gr.Markdown("#### 预测开关（与声学配置保持一致）")
        with gr.Row():
            pred_dur = gr.Checkbox(value=True, label="predict_dur（音素长度）")
            pred_pitch = gr.Checkbox(value=True, label="predict_pitch（音高）")
            pred_eng = gr.Checkbox(value=False, label="predict_energy")
            pred_bre = gr.Checkbox(value=False, label="predict_breathiness")
            pred_voi = gr.Checkbox(value=False, label="predict_voicing")
            pred_ten = gr.Checkbox(value=False, label="predict_tension")

        gr.Markdown("#### 参数范围（不建议随意修改）")
        with gr.Row():
            eng_min = gr.Number(value=-96.0, label="energy_db_min")
            eng_max = gr.Number(value=-12.0, label="energy_db_max")
            bre_min = gr.Number(value=-96.0, label="breathiness_db_min")
            bre_max = gr.Number(value=-20.0, label="breathiness_db_max")
            voi_min = gr.Number(value=-96.0, label="voicing_db_min")
            voi_max = gr.Number(value=-12.0, label="voicing_db_max")
            ten_min = gr.Number(value=-10.0, label="tension_logit_min")
            ten_max = gr.Number(value=10.0, label="tension_logit_max")

        gr.Markdown("#### 高级功能")
        with gr.Row():
            use_melody = gr.Checkbox(value=False, label="use_melody_encoder（旋律编码器）")
            melody_hidden = gr.Number(value=128, label="melody_hidden_size")
            use_glide = gr.Checkbox(value=False, label="use_glide_embed（滑音，需标注）")
            glide_types = gr.Textbox(value="up,down", label="glide_types（逗号分隔）")
            glide_scale = gr.Number(value=11.3137, label="glide_embed_scale（默认即可）")

        gr.Markdown("#### 字典与声码器配置")
        with gr.Row():
           dictionaries_config = gr.JSON(value={"zh": "dictionaries/opencpop-extension.txt"}, label="字典配置（JSON）")
           extra_phonemes_list = gr.Textbox(value="", label="额外音素（逗号分隔）")
           merged_phonemes_groups = gr.Textbox(value="", label="合并音素组（JSON 列表）")

        with gr.Row():
           vocoder_type = gr.Textbox(value="NsfHifiGAN", label="声码器类型")
           vocoder_ckpt = gr.Textbox(value="checkpoints/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt", label="声码器 ckpt")

        with gr.Row():
           use_lang = gr.Checkbox(value=False, label="使用多语言")
           num_lang = gr.Number(value=1, precision=0, label="语言数")
           use_spk = gr.Checkbox(value=False, label="使用多说话人")
           num_spk = gr.Number(value=1, precision=0, label="说话人数")




        # 数据集表格（与声学共用结构）
        gr.Markdown("#### 数据集列表（与声学配置共用结构）")
        with gr.Row():
            add_btn = gr.Button(" 添加行", scale=1)
            del_btn = gr.Button(" 删除末行", scale=1)
        ds_df = gr.Dataframe(
            headers=["raw_data_dir", "speaker", "spk_id", "language", "test_prefixes"],
            datatype=["str", "str", "number", "str", "str"],
            value=[["data/xxx1/raw", "speaker1", 0, "zh", "wav1,wav2"]],
            row_count=(1, 20),
            interactive=True,
        )

        add_btn.click(
    lambda df: pd.concat([df, pd.DataFrame([["data/xxx/raw", "speaker", 0, "zh", "test"]], columns=df.columns)], ignore_index=True),
    inputs=ds_df,
    outputs=ds_df
)
        # 导出
        with gr.Row():
            out_prefix = gr.Textbox(value="my_var", label="导出文件名前缀（不含 .yaml）")
            export_btn = gr.Button(" 导出 new_xxx_variance.yaml", variant="primary")
            export_log = gr.Textbox(label="日志", interactive=False)

#事件绑定
    export_btn.click(
            export_variance_config,
            inputs=[
                out_prefix, ds_df, lr, bs, max_step,
                pe, pe_ckpt, hnsep_ckpt,
                pred_dur, pred_pitch, pred_eng, pred_bre, pred_voi, pred_ten,
                eng_min, eng_max, bre_min, bre_max, voi_min, voi_max, ten_min, ten_max,
                use_melody, melody_hidden,
                use_glide, glide_types, glide_scale,
                out_prefix,
            ],
            outputs=export_log,
        )
        
    export_btn.click(
    export_variance_config,
    inputs=[
        out_prefix, ds_df, lr, bs, max_step,
        pe, pe_ckpt, hnsep_ckpt,
        pred_dur, pred_pitch, pred_eng, pred_bre, pred_voi, pred_ten,
        eng_min, eng_max, bre_min, bre_max, voi_min, voi_max, ten_min, ten_max,
        use_melody, melody_hidden,
        use_glide, glide_types, glide_scale,
        out_prefix,
        dictionaries_config,
        extra_phonemes_list,
        merged_phonemes_groups,
        vocoder_type,
        vocoder_ckpt,
        use_lang,
        num_lang,
        use_spk,
        num_spk,
    ],
    outputs=export_log,
)


        # 启动时检测模板
    def on_load():
            cfg, msg = load_var_template()
            if cfg is None:
                return msg, ""
            return msg, cfg.get("binary_data_dir", "").replace("/binary", "")[5:]
    blk.load(on_load, outputs=[load_st, model_name])

    return blk

# ---------- 全局路径 ----------
WORKSPACE = Path("/workspace/")
SOFA_BASE = WORKSPACE / "SOFA"
SOME_BASE = WORKSPACE / "SOME"
MDS_BASE  = WORKSPACE / "MakeDiffSinger"

# ---------- 事件 ----------
def set_workspace(path: str):
    global WORKSPACE, SOFA_BASE, SOME_BASE, MDS_BASE
    p = Path(path)
    if not p.is_dir():
        return "  目录不存在"
    WORKSPACE = p
    SOFA_BASE = WORKSPACE / "SOFA"
    SOME_BASE = WORKSPACE / "SOME"
    MDS_BASE  = WORKSPACE / "MakeDiffSinger"
    return f" 工作目录已设为：{WORKSPACE}"

# SOFA
def sofa_align(model: str, dictionary: str, fmt: str, save_conf: bool):
    if not (model and dictionary):
        return "  模型/词典路径为空"
    cmd = f'cd "{SOFA_BASE}" && python infer.py --ckpt "{model}" --dictionary "{dictionary}" --out_formats {fmt}'
    if save_conf:
        cmd += " --save_confidence"
    return run_cmd(cmd)

# SOME
def some_extract(model: str, dataset: str, overwrite: bool, enh: bool):
    if not (model and dataset):
        return "  模型/数据集路径为空"
    cmd = f'cd "{SOME_BASE}" && python batch_infer.py --model "{model}" --dataset "{dataset}"'
    if overwrite:
        cmd += " --overwrite"
    if enh:
        cmd += " --use_rmvpe"
    return run_cmd(cmd, timeout=1200)

# MDS
def run_labels_with_img(audio_dir: str, dict_path: str):
    if not (audio_dir and dict_path):
        return "  路径为空", None
    p = Path(audio_dir)
    if not p.is_dir():
        return f"  必须传入目录：{audio_dir}", None
    log = run_cmd(
        f'cd "{MDS_BASE}/acoustic_forced_alignment" && python validate_labels.py --dir "{audio_dir}" --dictionary "{dict_path}"',
        60
    )
    img_path = p / "phoneme_distribution.jpg"
    return log, (str(img_path) if img_path.exists() else None)

def run_pitch_with_img(wav_dir: str, tg_dir: str):
    if not (wav_dir and tg_dir):
        return " 路径为空", None
    pw, pt = Path(wav_dir), Path(tg_dir)
    if not (pw.is_dir() and pt.is_dir()):
        return " 两个参数都必须是目录", None
    log = run_cmd(
        f'cd "{MDS_BASE}/acoustic_forced_alignment" && python summary_pitch.py --wavs "{wav_dir}" --tg "{tg_dir}"',
        120
    )
    img_path = pw / "pitch_distribution.jpg"
    return log, (str(img_path) if img_path.exists() else None)


# ---------- Gradio ----------
with gr.Blocks(title="DiffSinger WEBUI试验版") as demo:
    gr.Markdown("#  DiffSinger WEBUI试验版")

    with gr.Tab(" 工作目录"):
        ws_inp = gr.Textbox(value=str(WORKSPACE), label="工作目录", lines=1)
        ws_btn = gr.Button("应用", variant="primary")
        ws_state = gr.Textbox(value="就绪", label="状态", interactive=False)
        ws_btn.click(set_workspace, inputs=ws_inp, outputs=ws_state)

    with gr.Tab(" SOFA "):
        mdl = gr.Textbox(label="模型.ckpt 绝对路径", placeholder="/SOFA/ckpt/xxx.ckpt")
        dic = gr.Textbox(label="词典.txt 绝对路径", placeholder="/SOFA/dictionary/xxx.txt")
        fmt = gr.Radio(["textgrid", "lab"], value="textgrid", label="输出格式")
        conf = gr.Checkbox(value=True, label="保存置信度")
        btn = gr.Button("开始对齐", variant="primary")
        log = gr.Textbox(label="输出", lines=10, interactive=False)
        btn.click(sofa_align, inputs=[mdl, dic, fmt, conf], outputs=log)

    with gr.Tab(" SOME "):
        mdl = gr.Textbox(label="模型.ckpt 绝对路径", placeholder="/SOME/pretrained/xxx.ckpt")
        dst = gr.Textbox(label="数据集目录（含wav）", placeholder="/dataset")
        ow = gr.Checkbox(value=True, label="覆盖已有")
        enh = gr.Checkbox(value=False, label="RMVPE 增强")
        btn = gr.Button("开始提取", variant="primary")
        log = gr.Textbox(label="输出", lines=10, interactive=False)
        btn.click(some_extract, inputs=[mdl, dst, ow, enh], outputs=log)

                        # ===== 4. 训练 =====
    with gr.TabItem("训练"):
        build_train_tab() 




    with gr.Tab(" MakeDiffSinger"):
        gr.Markdown("#### 1. 校验音频长度")
        d1 = gr.Textbox(placeholder="/segments")
        b1 = gr.Button("运行")
        o1 = gr.Textbox(lines=4, interactive=False)
        b1.click(lambda x: find_and_run_script("validate_lengths.py", MDS_BASE, f'--dir "{x}"', 30) if x else "   路径为空", inputs=d1, outputs=o1)
        # --- 2. 校验标签+字典 ---
        with gr.Group():
            gr.Markdown("### 2. validate_labels.py")
            with gr.Row():
                lab_dir = gr.Textbox(label="音频文件夹路径（含wav+lab）", placeholder="/segments")
                lab_dict = gr.Textbox(label="dictionary.txt 绝对路径", placeholder="/dictionary.txt")
            lab_btn = gr.Button("运行", variant="primary")
            lab_out = gr.Textbox(label="输出日志", interactive=False, lines=6)
            # 
            lab_img = gr.Image(label="phoneme_distribution.jpg", type="filepath", interactive=False)
            lab_btn.click(run_labels_with_img, inputs=[lab_dir, lab_dict], outputs=[lab_out, lab_img])

        # --- 3. 音高统计 ---
        with gr.Group():
            gr.Markdown("### 3. summary_pitch.py")
            with gr.Row():
                seg_wav = gr.Textbox(label="音频文件夹路径", placeholder="/segments")
                seg_tg  = gr.Textbox(label="TextGrid文件夹路径", placeholder="/textgrids")
            sum_btn = gr.Button("运行", variant="primary")
            sum_out = gr.Textbox(label="输出日志", interactive=False, lines=6)
            # 
            sum_img = gr.Image(label="pitch_distribution.jpg", type="filepath", interactive=False)
            sum_btn.click(run_pitch_with_img, inputs=[seg_wav, seg_tg], outputs=[sum_out, sum_img])

        gr.Markdown("#### 4. 构建训练集")
        d4a = gr.Textbox(placeholder="/segments")
        d4b = gr.Textbox(placeholder="/textgrids")
        d4c = gr.Textbox(placeholder="/xxx/raw")
        b4 = gr.Button("运行", variant="primary")
        o4 = gr.Textbox(lines=6, interactive=False)
        b4.click(lambda a, b, c: mds_script("/workspace/MakeDiffSinger/acoustic_forced_alignment/build_dataset.py", f'--wavs "{a}" --tg "{b}" --dataset "{c}"', 300) if all([a, b, c]) else "  路径为空", inputs=[d4a, d4b, d4c], outputs=o4)

        # ===================== 5. 二分式字典 =====================
        gr.Markdown("#### 5. 二分式字典（汉语/日语等）")
        csv_5 = gr.Textbox(label="1. transcriptions.csv 绝对路径", placeholder="/xxx/raw/transcriptions.csv")
        dict_5 = gr.Textbox(label="2. dictionary.txt 绝对路径", placeholder="/dictionary.txt")
        btn_5 = gr.Button("运行（二分式）", variant="primary")
        out_5 = gr.Textbox(lines=6, interactive=False)
        btn_5.click(lambda c, d: run_cmd(
            f'cd "{MDS_BASE}/variance-temp-solution" && python add_ph_num.py "{c}" --dictionary "{d}"', 180) if c and d else "  路径为空",
            inputs=[csv_5, dict_5], outputs=out_5)
        # ===================== 6. 单音节音素系统 =====================
        gr.Markdown("#### 6. 单音节音素系统（粤语/韩语等）")
        gr.Markdown("**步骤**：① 将所有元音写入 vowels.txt（空格分隔）；② 将所有辅音写入 consonants.txt（空格分隔）；③ 运行下方命令")
        csv_6 = gr.Textbox(label="1. transcriptions.csv 绝对路径", placeholder="/xxx/raw/transcriptions.csv")
        vow_6 = gr.Textbox(label="2. vowels.txt 绝对路径", placeholder="/vowels.txt")
        con_6 = gr.Textbox(label="3. consonants.txt 绝对路径", placeholder="/consonants.txt")
        btn_6 = gr.Button("运行（单音节）", variant="primary")
        btn_6.click(lambda c, v, co: run_cmd(
            f'cd "{MDS_BASE}/variance-temp-solution" && python add_ph_num.py "{c}" --vowels "{v}" --consonants "{co}"', 180) if all([c, v, co]) else "  路径为空",
            inputs=[csv_6, vow_6, con_6], outputs=out_5)
        # ===================== 7. 多音节音素系统 =====================
        gr.Markdown("#### 7. 多音节音素系统（英语/俄语等）")
        gr.Markdown("**步骤**：① 将所有元音写入 vowels.txt（空格分隔）；② 将所有辅音写入 consonants.txt（空格分隔）；③ 将所有流音写入 liquids.txt（空格分隔，可无）；④ 运行下方命令")
        csv_7 = gr.Textbox(label="1. transcriptions.csv 绝对路径", placeholder="/xxx/raw/transcriptions.csv")
        tg_7  = gr.Textbox(label="2. TextGrid 文件夹绝对路径", placeholder="/xxx/raw/textgrids")
        vow_7 = gr.Textbox(label="3. vowels.txt 绝对路径", placeholder="/vowels.txt")
        con_7 = gr.Textbox(label="4. consonants.txt 绝对路径", placeholder="/consonants.txt")
        liq_7 = gr.Textbox(label="5. liquids.txt 绝对路径（可空）", placeholder="/liquids.txt")
        btn_7 = gr.Button("运行（多音节）", variant="primary")
        btn_7.click(lambda c, tg, v, co, l: run_cmd(
            f'cd "{MDS_BASE}/variance-temp-solution" && python add_ph_num_advanced.py "{c}" --tg "{tg}" --vowels "{v}" --consonants "{co}"' + (f' --liquids "{l}"' if l else ""), 180) if all([c, tg, v, co]) else "  路径为空",
            inputs=[csv_7, tg_7, vow_7, con_7, liq_7], outputs=out_5)
                        # ===== 5. ONNX 导出 =====
    with gr.TabItem("ONNX 导出"):
        build_onnx_tab()
    






    
    
    with gr.Tab("配置文件"):
        with gr.Tabs():
            # ===== 1. 声学配置 =====
            with gr.TabItem("声学配置（只能看看，实际上无法使用）"):
                build_acoustic_subtab()     
            # ===== 2. 唱法配置 =====
            with gr.TabItem("唱法配置（只能看看，实际上无法使用）"):
                build_variance_subtab()
            # ===== 3. 文本编辑器 =====
            with gr.TabItem("配置文本编辑器"):
                build_mini_editor()






demo.launch(
    server_name="0.0.0.0", 
    server_port=7861, 
    share=True,
    allowed_paths=["/"]
)