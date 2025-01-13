import uvicorn
# -*- coding:utf-8 -*-
from fastapi import FastAPI
from XT_run_glm import gpt_answer
import socket

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/ask/{question}")
async def answer_question(question):
    answer = gpt_answer(question)
    print("Q=" + question + ", answer = " + answer)
    return {"我:": question, "小白白GPT:":answer}

@app.get("/a/{question}")
async def answer_question(question):
    answer = gpt_answer(question)
    print("Q=" + question + ", answer = " + answer)
    return answer

def get_my_ipv4():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return ip


if __name__ == "__main__":
    ip = get_my_ipv4()
    print("my ip = " + ip)
    uvicorn.run(app, host=ip, port=8000, workers=1)
