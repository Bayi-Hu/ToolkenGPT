import os

os.chdir("..")
from funchub.math import custom_round
import re
import json


def parse_answer(answer, pattern: str = "####"):
    if pattern == "####":
        answer = answer.split("####")[-1]
        answer = answer.strip().strip("\n").strip('\\n')
        # 32,333 -> 32333
        answer = answer.replace(",", "")

        # get the last number
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0
    elif pattern == "answer is":
        answer = answer.split("answer is")[-1]
        answer = answer.strip().strip("\n").strip('\\n')

        # 32,333 -> 32333
        answer = answer.replace(",", "")

        # get the last number
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0

    return answer


def accuracy(pred, true, type="exact"):
    if len(pred) < len(true):
        true = true[:len(pred)]

    correct = 0
    for p, t in zip(pred, true):
        try:
            if type == "exact":
                if float(p) == float(t):
                    correct += 1
            elif type == "round":
                if round(float(p), 2) == custom_round(float(t), 2):
                    correct += 1
            elif type == "approx":
                # 1% error tolerance, e.g. 1000 -> 990 ~ 1010
                if abs(float(p) - float(t)) <= abs(float(t)) * 0.001:
                    correct += 1
        except ValueError:
            pass

    return correct / len(pred)


if __name__ == '__main__':

    target_path = "./data/funcqa/funcqa_oh.json"
    eval_path = "./outputs/funcqa_oh/llama-2-13b_funcqa_0.0001_5_func_embedding_2.7.jsonl"

    with open(target_path, "r") as f:
        groundtruth_data = json.load(f)

    answer = [d["answer"] for d in groundtruth_data]

    with open(eval_path, "r") as f:
        pred_data = [json.loads(line) for line in f]

    pred = [parse_answer(d["generation"], pattern="####") for d in pred_data]

    print(pred)
    print(answer[:len(pred)])

    print("Accuracy: ", accuracy(pred, answer[:len(pred)], type="approx"))

    # dummmy_answer = ['32.3799', '43', '100', '1556750']
    # print("Accuracy: ", accuracy(dummmy_answer, answer[:len(dummmy_answer)], type="approx"))