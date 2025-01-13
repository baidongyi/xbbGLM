import os.path

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

import sys
import os
from chat_history_tool import *

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

def restart():
    python = sys.executable
    os.execl(python, python, *sys.argv)

app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    user_name = json_post_list.get('user_name')

    #如果没有发送历史过来，则获取本地历史
    if history is None and user_name is not None:
        history = get_my_history(user_name, get_row_to_load())
    else:
        _ = get_my_history(user_name, get_row_to_load())

    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()

    update_history_file(user_name, prompt, response)

    return answer


if __name__ == '__main__':
    model_path = r"C:\Users\administrator\xxtony\xbbgpt\chatglm-6b-int4"
    if not os.path.exists(model_path):
        model_path = r"C:\Users\baido\OneDrive\Work\AI\xbbGPT\chatglm-6b-int4"


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
