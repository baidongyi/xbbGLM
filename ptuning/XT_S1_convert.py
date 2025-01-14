import shutil, os

import pandas as pd

from XT_S0_parameters import project_root_path

def escape_str(text:str):
    return str(str(str(text).replace("\n"," ")).replace("\"","'")).replace("“","'")


def get_lines_from_file(file_path: str, max_row: int):
    df = pd.read_excel(file_path)

    print('reading data from file_path=' + file_path)

    lines = []

    if df['Q'].size > max_row:
        my_max_row = max_row
    else:
        my_max_row = df['Q'].size

    for i in range(0, my_max_row):
        q, a = df['Q'].iloc[i], df['A'].iloc[i]
        line = "{\"content\": \"" + escape_str(q) + "\", \"summary\": \"" + escape_str(a) + "\"}\n"
        lines.append(line)

    return lines


def get_all_lines_from_folder(folder_path: str, max_row: int):
    files = os.listdir(folder_path)
    result = []
    for one_file in files:
        full_path = folder_path + "//" + one_file
        lines = get_lines_from_file(full_path, max_row)
        result += lines
    return result


def write_lines_into_file(lines: [], dest_file: str):
    with open(dest_file, 'w', encoding='utf-8') as file:
        file.writelines(lines)


if __name__ == '__main__':
    source_folder = project_root_path + "\\ChatGLM-6B-main\\ptuning\\source_files"
    dest_file = project_root_path + "\\ChatGLM-6B-main\\ptuning\\dest_files\\d1.json"
    val_file_path = project_root_path + "\\ChatGLM-6B-main\\ptuning\\dest_files\\t1.json"

    lines = get_all_lines_from_folder(source_folder, max_row=99)
    write_lines_into_file(lines, dest_file)

    shutil.copyfile(dest_file, val_file_path)
