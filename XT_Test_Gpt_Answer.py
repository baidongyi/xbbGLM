import os

from XT04_run_glm_save_history import gpt_answer

if __name__ == '__main__':
    for i in range(0, 30):
        ask = input("new question input:")
        user_name = os.getlogin()
        answer = gpt_answer(user_name + ':' + ask)
        print(str(i) + ". ask:" + ask + "\nanswer:" + answer)