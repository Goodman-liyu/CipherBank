from turtle import mode
from prompt import *
from utils.tools import *
import argparse


def test(data, cipher_type, model, model_name, shot_number=3, is_hint=False, number=False):

    results = []
    if is_hint == True:
        output_file = f"./predictions/{shot_number}-shot-hint/{args.model}/{cipher_type}.json"
        log_file = f"./log/{shot_number}-shot-hint/{args.model}/{cipher_type}.log"
    elif number == True:
        output_file = f"./predictions/{shot_number}-all/{args.model}/{cipher_type}.json"
        log_file = f"./log/{shot_number}-all/{args.model}/{cipher_type}.log"
    else:
        output_file = f"./predictions/{shot_number}-shot-letter/{args.model}/{cipher_type}.json"
        log_file = f"./log/{shot_number}-shot-letter/{args.model}/{cipher_type}.log"

    # output_file = f"./predictions/normal/{args.model}/{cipher_type}_normal.json"
    # log_file = f"./log/normal/{args.model}/{cipher_type}_normal.log"

    correct = 0
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        open(log_file, "w").close()
    levenshtein_sims = []
    for i, item in tqdm(enumerate(data), total=len(data), desc="Decryption Progress"):
        system_prompt, user_prompt = get_prompt(
            format_type="markdown",
            cipher_type=cipher_type,
            few_shot=shot_number,
            ciphertext=item[cipher_type],
            is_hint=is_hint,
            number=number,
        )

        print(system_prompt)
        print(user_prompt)

        if "o1" in model_name:
            response = model(prompts=user_prompt)
        elif "gemini" in model_name or "claude" in model_name:
            response = model(model_name, system_prompt, user_prompt)
        else:
            response = model(system_prompt=system_prompt, prompts=user_prompt)
        sentence = extract_result_sentences(response)

        is_correct = compare_strings(sentence, item["plaintext"])
        levenshtein_sim = levenshtein_similarity(sentence, item["plaintext"])
        levenshtein_sims.append(levenshtein_sim)
        result = {
            "details": {
                str(i): {
                    "prompt": [
                        {"role": "SYSTEM", "prompt": system_prompt},
                        {"role": "HUMAN", "prompt": user_prompt},
                    ],
                    "origin_prediction": response,
                    "predictions": sentence,
                    "references": item["plaintext"],
                    "is_correct": is_correct,
                    "levenshtein_similarity": levenshtein_sim,
                }
            }
        }
        correct += result["details"][str(i)]["is_correct"]
        results.append(result)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Processing the {i+1}th data...\n")
            f.write(f"Ciphertext: {item[cipher_type]}\n")
            f.write(f"Decryption result: {sentence}\n")
            f.write(f"Reference answer: {item['plaintext']}\n")
            f.write(f"Is correct: {is_correct}\n")
            f.write(f"Levenshtein similarity: {levenshtein_sim}\n")
            f.write(f"-----------------------------------\n")

    total = len(results)
    accuracy = (correct / total) * 100
    levenshtein_sim_avg = sum(levenshtein_sims) / len(levenshtein_sims)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_result = {"accuracy": accuracy, "levenshtein_similarity_avg": levenshtein_sim_avg, "details": {}}
    for i, result in enumerate(results):
        final_result["details"].update(result["details"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")


def main(args):
    data = read_jsonl(args.input_file)
    if args.model == "deepseek-r1":
        model = DeepSeek_R1()
    elif args.model == "deepseek-v3":
        model = DeepSeek_V3()
    elif args.model == "qwq":
        model = QwQLLM()
    elif "gemini" in args.model or "claude" in args.model:
        model = Agent()
    else:
        model = None#OpenAILLM(model_name=args.model)
    test(data, args.cipher_type, model, args.model, args.shot_number, args.is_hint, args.number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="data/test.jsonl",
        type=str,
        help="Path to the input file to be processed",
    )
    parser.add_argument("--cipher_type", default="Rot13", type=str, help="Type of encryption algorithm")
    parser.add_argument("--number", action="store_true", help="Whether to encrypt numbers")
    parser.add_argument("--is_hint", action="store_true", help="Whether to provide hints")
    parser.add_argument("--shot_number", default=3, type=int, help="Number of few-shot attempts")
    parser.add_argument(
        "--model",
        default="o1-mini",
        type=str,
        choices=[
            "deepseek-r1",
            "qwq",
            "o1-mini",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "o1-preview-2024-09-12",
            "deepseek-v3",
            "MiniMaxText01",
            "claude-3-5-sonnet-20241022",
            "gemini-2.0-flash-thinking-exp",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
            "o1-2024-12-17",
            "o1",
            "DeepSeek-V3",
            "DeepSeek-R1",
        ],
        help="Model name",
    )
    args = parser.parse_args()

    main(args)
