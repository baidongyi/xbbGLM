import os.path

from XT_read_qa_lib import *
from XT_excel_writer import insert_data_to_sheet

from shutil import copyfile
import os
import pandas as pd

import torch, gc


history_path = r"C:\Users\baido\OneDrive\Work\AI\xbbGPT\ChatGLM-6B-main\history_lib"



def create_history_file(new_file_path: str):
    if not os.path.exists(new_file_path):
        temp_file = history_path + '\\temp.xlsx'
        copyfile(temp_file, new_file_path)


def read_list_from_file(file_path: str, max_row: int):
    if not os.path.exists(file_path):
        return []

    df = pd.read_excel(file_path)

    print('--- reading data from file = ' + file_path)

    my_list = []

    if df['Q'].size > max_row:
        my_max_row = max_row
    else:
        my_max_row = df['Q'].size

    for j in range(1, my_max_row):
        q, a = df['Q'].iloc[j], df['A'].iloc[j]
        my_list.append((q, a))
        print(str(j) + ".\t Q:" + str(q) + "\t -> " + "  A:" + str(a) + " ")

    print('=== END reading data from file = ' + file_path)

    return my_list


def get_my_history(my_user_name: str, max_row: int):
    history_file_path = os.path.join(history_path, my_user_name) + ".xlsx"

    if os.path.exists(history_file_path):
        result_list = read_list_from_file(history_file_path, max_row)

    else:
        create_history_file(history_file_path)
        result_list = []

    prepared_content = read_list_from_file(os.path.join(history_path, 'all.xlsx'), 10)

    return result_list + prepared_content


def update_history_file(my_user_name: str, my_question: str, my_answer: str):
    history_file_path = os.path.join(history_path, my_user_name) + ".xlsx"
    insert_data_to_sheet(history_file_path, my_question, my_answer)


def get_row_to_load():
    return 15