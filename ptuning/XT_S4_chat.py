from XT_S0_parameters import para

from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import torch, gc


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def get_trained_model_token():
    clear_mem()

    model_path = para['model_path']
    check_point_path = para['train_checkpoint_path'] + "\\checkpoint-3000"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=para['pre_seq_len'])
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(check_point_path, "pytorch_model.bin"))

    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    # Comment out the following line if you don't use quantization
    model = model.quantize(4)  # 或者8
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()

    return model, tokenizer


if __name__ == "__main__":
    my_model, my_token = get_trained_model_token()
    history = []

    question = input(". 我问: ")

    response, history = my_model.chat(my_token, question, history=history)

    print("小白白: " + str(response))
