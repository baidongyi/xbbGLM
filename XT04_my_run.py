from transformers import AutoTokenizer, AutoModel
from XT00_parameter import para


def chat(prompt, history):
    path = para['model_path']

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, revision="")
    model = AutoModel.from_pretrained(path, trust_remote_code=True, revision="").half().cuda()
    model = model.eval()

    res, history = model.chat(tokenizer, prompt, history)
    return res


if __name__ == "__main__":
    ask = "我获得了诺贝尔奖，帮我写一个获奖谢词，谢谢祖国、父母和同事"
    response = chat(ask,[])
    print(response)
