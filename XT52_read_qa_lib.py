import pandas as pd
import os


def get_qa_list(file_name: str, max_row=20):
    file_path = file_name
    df = pd.read_excel(file_path)

    print('reading data from file=' + file_path)

    result = []

    if df['Q'].size > max_row:
        my_max_row = max_row
    else:
        my_max_row = df['Q'].size

    for i in range(0, my_max_row):
        q, a = df['Q'].iloc[i], df['A'].iloc[i]

        result.append((q, a))
        print(str(i) + ".-------\nQ:" + q + "\n->" + "A:" + a + "\n=========")

    return result


def get_qa_file_name(question: str):
    path = "../ChatGLM-6B-main/qa_lib"
    file_list = os.listdir(path)
    for file_name in file_list:
        index = file_name.index('.xlsx')
        keyword = file_name[:index].upper()
        print('keyword = ' + keyword)
        if keyword in question.upper():
            result = path + '/' + file_name
            print('result = ' + result)
            return result

    return ''


def load_qa_lib_by_question(question: str, max_row=20):
    file_name = get_qa_file_name(question)
    if file_name != "":
        qa_list = get_qa_list(file_name, max_row)
    else:
        qa_list = []

    return qa_list


if __name__ == "__main__":
    my_list = load_qa_lib_by_question("EX是什么东西?")
    print(my_list)
