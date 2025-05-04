import sys
import os

print("Initial sys.path:")
for i, p in enumerate(sys.path):
    print(f"{i}: {p}")

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'webapp/utils')
print(f"\nCalculated utils_dir: {utils_dir}")

if utils_dir not in sys.path:
    print(f"Appending {utils_dir} to sys.path")
    sys.path.append(utils_dir)
else:
    print(f"{utils_dir} is already in sys.path")

print("\nFinal sys.path:")
for i, p in enumerate(sys.path):
    print(f"{i}: {p}")

# ここで utils_dir 内のモジュールをインポートしてみる
# 例: import your_module_in_utils
# または: from your_package_in_utils import some_function
try:
    # インポートしたいモジュール名やパッケージ名に置き換えてください
    # import your_module_name
    print("\nAttempting to import module from utils_dir...")
    # 例: import my_utility_script # utils_dir/my_utility_script.py がある場合
    # 例: from my_utils_package import some_func # utils_dir/my_utils_package/__init__.py がある場合
    # import debug_module # テスト用のシンプルなモジュール名に置き換える
    import face_detect_crop
    print("Import successful or not executed.") # インポート文自体はここに直接書かず、このスクリプトを実行する親スクリプト等に書くのが一般的ですが、デバッグのためにここに書くことも可能
except ImportError as e:
    print(f"ImportError after appending path: {e}")

