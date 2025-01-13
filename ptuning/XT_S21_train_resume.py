
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
         "--logging_steps 25 " \
         "--save_steps 1000 " \
         "--learning_rate 1e-5 " \
         "--pre_seq_len 32 " \
         "--quantization_bit 8 " \
         "--resume_from_checkpoint output/" + para['checkpoint_name'] + "/checkpoint-3000"

    os.system(f"python main.py {option}")



if __name__ == "__main__":
    train()