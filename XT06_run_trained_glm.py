import warnings

from transformers import AutoTokenizer, AutoModel, AutoConfig
from XT52_read_qa_lib import *

from XT51_excel_writer import insert_data_to_sheet
from shutil import copyfile
import os
import pandas as pd

import torch, gc

from ptuning.XT_S0_parameters import para

root_path = ".."

path = "C:\\Users\\administrator\\xxtony\\xbbgpt\\chatglm-6b-int4"

history_path = root_path + "\\ChatGLM-6B-main\\history_lib"


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

    start_row = df['Q'].size - max_row

    if start_row <= 0:
        start_row = 0

    my_max_row = df['Q'].size

    print('start_row= ' + str(start_row) + ', my_max_row = ' + str(my_max_row))

    for j in range(start_row, my_max_row):
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

    prepared_content = read_list_from_file(os.path.join(history_path, 'all.xlsx'), 30)

    return result_list + prepared_content


def update_history_file(my_user_name: str, my_question: str, my_answer: str):
    history_file_path = os.path.join(history_path, my_user_name) + ".xlsx"
    insert_data_to_sheet(history_file_path, my_question, my_answer)


def get_video_size():
    return round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)


def get_row_to_load():
    size = get_video_size()
    print('vram size = ' + str(size))
    if size > 18:
        return 25
    else:
        return 15


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def get_trained_model_token():
    clear_mem()

    model_path = para['model_path']
    check_point_path = para['train_checkpoint_path'] + "\\checkpoint-3000"

    the_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
    the_model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(check_point_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    the_model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    # Comment out the following line if you don't use quantization
    the_model = the_model.quantize(4)  # 或者8
    the_model = the_model.half().cuda()
    the_model.transformer.prefix_encoder.float()
    the_model = the_model.eval()

    return the_model, the_tokenizer


if __name__ == "__main__":

    warnings.resetwarnings()
    warnings.simplefilter('ignore', SyntaxWarning)

    model, tokenizer = get_trained_model_token()

    user_name = os.getlogin()

    history = get_my_history(user_name, get_row_to_load())

    for i in range(1, 99):
        print("--------------------------")
        ask = input("请输入:")

        response, history = model.chat(tokenizer, ask, history=history)

        print(str(i) + "\t.我:" + ask)
        print(str(i) + "\t.GPT:" + response)

        update_history_file(user_name, ask, response)

        print("===========================")
        print('------ history --------')
        print(history)
        print('======= history ========')


def get_index(word: str):
    try:
        my_index = word.index(":")
        return my_index
    except:
        return -1


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def gpt_trained_answer(question: str):
    clear_mem()

    index = get_index(question)

    if index >= 0:
        my_user_name = question[:index]
        question = question[index + 1:]
    else:
        my_user_name = "any"

    row_max = get_row_to_load()
    my_history = get_my_history(my_user_name, row_max)

    my_model, my_tokenizer = get_trained_model_token()

    my_response, my_history = my_model.chat(my_tokenizer, question.replace(" ", ""), history=my_history,
                                            max_length=4666)

    update_history_file(my_user_name, question, my_response)

    return my_response
