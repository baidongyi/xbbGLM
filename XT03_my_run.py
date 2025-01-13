from transformers import AutoTokenizer, AutoModel


if __name__ == "__main__":
    path = "C:\\Users\\baido\\OneDrive\\Learning\\Project\\Python\\ChatGLM-6B\\chatglm-6b-int4"

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, revision="")
    model = AutoModel.from_pretrained(path, trust_remote_code=True, revision="").half().cuda()
    model = model.eval()
    ask = "我获得了诺贝尔奖，帮我写一个获奖谢词，谢谢祖国、父母和同事"
    response, history = model.chat(tokenizer, ask, history=[])
    print(response)

