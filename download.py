import os
import urllib.request
import time
import sys

def download_file(url, filename, max_retries=3):
    """
    下载文件并显示进度
    :param url: 文件URL
    :param filename: 本地保存文件名
    :param max_retries: 最大重试次数
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # 创建下载目录（如果不存在）
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 开始下载
            print(f"开始下载: {url}")
            print(f"保存到: {filename}")
            
            # 使用回调函数显示进度
            def report_progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r进度: {percent}% [{count * block_size / (1024 * 1024):.2f}MB/{total_size/(1024 * 1024):.2f}MB]")
                sys.stdout.flush()
            
            # 下载文件
            urllib.request.urlretrieve(url, filename, report_progress)
            
            # 检查文件大小
            file_size = os.path.getsize(filename)
            print(f"\n下载完成! 文件大小: {file_size/(1024 * 1024):.2f}MB")
            return True
            
        except Exception as e:
            retry_count += 1
            print(f"\n下载失败: {str(e)}")
            print(f"重试 {retry_count}/{max_retries}...")
            time.sleep(2)  # 等待2秒后重试
    
    print(f"下载失败，超过最大重试次数: {url}")
    return False

def main():
    # 文件下载列表
    download_list = [
        {
            "url": "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip",
            "filename": "vocoders/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"
        },
        {
            "url": "https://github.com/yxlllc/vocal-remover/releases/download/hnsep_240512/hnsep_240512.zip",
            "filename": "vocal_remover/hnsep_240512.zip"
        },
        {
            "url": "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip",
            "filename": "rmvpe/rmvpe.zip"
        },
        {
            "url": "https://github.com/BaiShuoQwQ/SOFA_Models/releases/download/v2.0/SOFA_model_mandarin_byBaiShuo.zip",
            "filename": "sofa_models/SOFA_model_mandarin_byBaiShuo.zip"
        },
        {
            "url": "https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip",
            "filename": "some_models/0119_continuous128_5spk.zip"
        }
    ]
    
    print("=" * 50)
    print("开始批量下载文件")
    print("=" * 50)
    
    # 依次下载每个文件
    for i, item in enumerate(download_list):
        print(f"\n[{i+1}/{len(download_list)}]")
        success = download_file(item["url"], item["filename"])
        
        if not success:
            print("跳过此文件，继续下载下一个...")
    
    print("\n" + "=" * 50)
    print("所有文件下载完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()