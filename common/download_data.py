"""
下载 text8 数据集
用法: python common/download_data.py
"""
import os
import zipfile
import urllib.request


def download_text8(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, 'text8')

    if os.path.exists(target):
        print(f'[download] text8 已存在: {target}')
        return target

    url = 'http://mattmahoney.net/dc/text8.zip'
    zip_path = os.path.join(data_dir, 'text8.zip')

    print(f'[download] 正在下载 {url} ...')
    urllib.request.urlretrieve(url, zip_path)

    print('[download] 解压中 ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)

    os.remove(zip_path)
    print(f'[download] 完成: {target}')
    return target


if __name__ == '__main__':
    download_text8()
