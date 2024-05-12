import os
import codecs

def convert_encoding(file_path, from_encoding, to_encoding='utf-8'):
        try:
            with codecs.open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                f.read()
            return True
        except UnicodeDecodeError:
            print(f"*********{file_path}")
            try:
                with open(file_path, 'r', encoding=from_encoding, errors='ignore') as f:
                    content = f.read()
                with open(file_path, 'w', encoding=to_encoding) as f:
                    f.write(content)
                print(f"Converted {file_path} from {from_encoding} to {to_encoding}")        
            except Exception as e:
                print(f"Error converting {file_path}: {e}")

def convert_files_to_utf8(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            convert_encoding(file_path, 'gbk')

# 指定目录路径
directory_path = "/home/feixuwu/MyCode/llama2.c/data/simplestory"

# 调用函数转换所有文件编码为UTF-8
convert_files_to_utf8(directory_path)

