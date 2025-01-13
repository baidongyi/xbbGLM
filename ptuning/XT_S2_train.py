
import os
from XT_S0_parameters import para

def train():
    option = "--do_train " \
         "--train_file " + para['train_file'] + " " \
         "--validation_file " + para['val_file'] + " " \
         "--prompt_column content " \
         "--response_column summary " \
         "--overwrite_cache " \
         "--model_name_or_path " + para['model_path'] + " " \
         "--output_dir output/" + para['checkpoint_name'] + " " \
         "--overwrite_output_dir " \
         "--max_source_length 128 " \
         "--max_target_length 256 " \
         "--per_device_train_batch_size 4 " \
         "--per_device_eval_batch_size 4 " \
         "--gradient_accumulation_steps 16 " \
         "--predict_with_generate " \
         "--max_steps " + str(para['max_step'])  + " " \
         "--logging_steps 100 " \
         "--save_steps 1000 " \
         "--learning_rate " + str(para['learning_rate']) + " " \
         "--pre_seq_len " + str(para['pre_seq_len']) + " " \
         "--quantization_bit 4 "

    os.system(f"python main.py {option}")



if __name__ == "__main__":
    train()