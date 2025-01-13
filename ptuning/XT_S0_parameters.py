para = {}

project_root_path = "C:\\Users\\baido\\OneDrive\\Work\\AI\\xbbGPT"

para['model_path'] = project_root_path + "\\chatglm-6b-int4"

para['train_file'] = project_root_path + "\\ChatGLM-6B-main\\ptuning\\dest_files\\t1.json"
para['val_file'] = project_root_path + "\\ChatGLM-6B-main\\ptuning\\dest_files\\d1.json"

para['checkpoint_name'] = 'adgen-chatglm-6b-pt-20241228'
para['train_checkpoint_path'] = project_root_path + "\\ChatGLM-6B-main\\ptuning\\output\\" + para['checkpoint_name']

para['max_step'] = 3000
para['learning_rate'] = '1E-5'
para['pre_seq_len'] = 128

if __name__ == '__main__':
    print(para)
