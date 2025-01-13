
import os
from XT_S0_parameters import para

def eval():

    check_point = "3000"

    option = "--do_predict " \
     "--validation_file "+ para['val_file'] + " " \
     "--test_file "+ para['val_file'] + " " \
    "--overwrite_cache " \
    "--prompt_column content " \
    "--response_column summary " \
    "--model_name_or_path " + para['model_path'] + " " \
    "--ptuning_checkpoint " +para['train_checkpoint_path'] + "\\checkpoint-" + str(check_point) + " " \
    "--output_dir ./output/" + para['checkpoint_name'] + "\\checkpoint-" + str(check_point) + " " \
    "--overwrite_output_dir " \
    "--max_source_length 128 " \
    "--max_target_length 256 " \
    "--per_device_eval_batch_size 4 " \
    "--predict_with_generate " \
    "--pre_seq_len " + str(para['pre_seq_len']) + " " \
    "--quantization_bit 4 "

    os.system(f"python main.py {option}")



if __name__ == "__main__":
    eval()