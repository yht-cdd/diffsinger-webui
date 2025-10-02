
基于https://github.com/openvpi/DiffSinger
实现的一种方式

1克隆仓库
```
git clone https://github.com/openvpi/MakeDiffSinger.git
git clone https://github.com/qiuqiao/SOFA.git
git clone https://github.com/openvpi/SOME.git
git clone https://github.com/openvpi/DiffSinger.git
```
2安装模块
```
#安装pytorch
pip3 install torch torchvision
#安装tqdm
pip install tqdm
#安装ffmpeg
pip install ffmpeg
#或者
sudo apt install ffmpeg
```
安装依赖
```

pip install -r MakeDiffSinger/acoustic_forced_alignment/requirements.txt
pip install -r MakeDiffSinger/variance-temp-solution/requirements.txt
pip install -r SOFA/requirements.txt 
pip install -r SOME/requirements.txt 
pip install -r DiffSinger/requirements.txt 
```
ONNX导出相关内容
```
conda create -n ONNX python=3.8 -y
conda activate ONNX
pip install -r DiffSinger/requirements-onnx.txt 
#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```
如果SOME报错，你需要
```
python -m pip install "pip<24.1"
pip install -r SOME/requirements.txt
```
关于webui,试验版webui,可能在一定情况下有用
示例参考<img width="1920" height="1080" alt="屏幕截图 2025-10-01 230739" src="https://github.com/user-attachments/assets/4991cbef-39bd-40c3-843f-3d5f9b14ff2b" />





