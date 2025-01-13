
import requests



def get_url():
    # return r"http://10.1.24.2:19980/glm/"
    return "http://127.0.0.1:8000"

def chatGLM(prompt, history, user_name):
    resp = requests.post(
        url=get_url(),
        json={"prompt": prompt, "history": history, "user_name":user_name},
        headers={"Content-Type": "application/json;charset=utf-8"}
    )
    return resp.json()['response'], resp.json()['history']

if __name__ == '__main__':
    history = []
    question = "hi"
    user_name = "baido"
    response, history = chatGLM(question, history, user_name)
    print('Answer:', response)
