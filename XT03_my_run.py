from transformers import AutoTokenizer, AutoModel
import torch

from XT00_Parameter import para

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()


def chat(prompt, history):
    torch_gc()

    path = para['model_path']

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, revision="")
    model = AutoModel.from_pretrained(path, trust_remote_code=True, revision="").half().cuda()
    model = model.eval()
    res, history = model.chat(tokenizer, prompt, history=history)
    return res


if __name__ == "__main__":

    ask = input("你问:")
    response = chat(ask, [])
    print(f"GLM回答:{response}")

