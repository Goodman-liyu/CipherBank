import json

def get_prompt(
    format_type="markdown",
    cipher_type=None,
    few_shot=3,
    ciphertext=None,
    is_hint=False,
    number=False,
):

    system_format_funcs = {
        "markdown": get_prompt_system_markdown,
    }

    user_format_funcs = {
        "markdown": get_prompt_user_markdown,
    }

    if format_type not in system_format_funcs:
        raise ValueError(f"Unsupported format type: {format_type}")

    system_prompt = system_format_funcs[format_type]()
    user_prompt = user_format_funcs[format_type](cipher_type, few_shot, ciphertext, is_hint, number)

    return system_prompt, user_prompt

def get_prompt_system_markdown():

    begin = """- Role: Cryptography Analysis Expert.
    - Goals: Utilize the provided ciphertext and plaintext examples to analyze encryption patterns, decrypt new ciphertext.
    - Workflow:
    1. Analyze the provided ciphertext and plaintext examples to identify possible encryption patterns and rules.
    2. Apply the decryption algorithm to the new ciphertext, attempt to decrypt, and verify the results."""
    # begin = "You are a cryptography expert, please decrypt according to the example provided by the user"
    return begin


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing error: {e}")
                    continue
    return data

data = read_jsonl("data/shot_case.jsonl")

def get_prompt_user_markdown(cipher_type, few_shot, ciphertext, is_hint=False, number=False):

    if is_hint:
        begin = f"""## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given algorithm flow and examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Algorithm Flow:
    {Introduce(cipher_type)}
## Examples:"""
    else:
        begin = f"""## Background:
    Decrypt the provided **Ciphertext** and return the corresponding **Plaintext** following the given examples.
    Think step by step.
    Provide the **Plaintext** result in the format <result>text</result>, where text is the decrypted text.
## Examples:"""

    case = ""

    for i in range(few_shot):
        case += f"""
    * Example {i+1}:
        - **Ciphertext**:
            {data[i][cipher_type]}
        - **Plaintext**:
            {data[i]["plaintext"]}"""
    if isinstance(ciphertext, list):
        question = f"""
## Input:"""
        for x in ciphertext:
            question += f"""
    - **Ciphertext**:
        {x}
    - **Plaintext**:      
"""
    else:
        question = f"""
## Input:
    - **Ciphertext**:
        {ciphertext}
    - **Plaintext**:
"""
    return begin + case + question


def Introduce(cipher_type):
    if cipher_type == "Rot13":
        return """Uses the Caesar cipher with a fixed shift of 13 positions. For each letter in the **Plaintext**, shift it forward by 13 positions in the alphabet to produce the **Ciphertext**."""
    if cipher_type == "Atbash":
        return """Uses the Atbash cipher. Each letter in the **Plaintext** is replaced with its reverse counterpart in the alphabet."""
    elif cipher_type == "Vigenere":
        return """Uses the Vigenère cipher. Each letter in the **Plaintext** is shifted by the corresponding letter in the **Key** to produce the **Ciphertext**."""
    elif cipher_type == "Polybius":
        return """Uses the Polybius cipher. Each letter in the **Plaintext** is mapped to a pair of coordinates in the Polybius square, forming the **Ciphertext**."""
    elif cipher_type == "Reverse":
        return """Reverses the **Plaintext** to create the **Ciphertext**."""
    elif cipher_type == "SwapPairs":
        return """For each pair of letters in the **Plaintext**, their positions are swapped to produce the **Ciphertext**. If the number of letters is odd, the last letter remains in its original position."""
    elif cipher_type == "rail_fence":
        return """Uses the Rail Fence cipher. The **Plaintext** is written in a zigzag pattern along multiple "rails", then read off row by row to form the **Ciphertext**."""
    elif cipher_type == "DualAvgCode":
        return """This encryption method converts each letter of the **Plaintext** into two letters in the **Ciphertext**, such that the average of their ASCII values equals the ASCII value of the original letter."""
    elif cipher_type == "ParityShift":
        return """For each letter in the **Plaintext**: 
        - If the ASCII value is even, add 1 to it to get the corresponding character in the **Ciphertext**.
        - If the ASCII value is odd, subtract 1 to get the new character in the **Ciphertext**."""
    elif cipher_type == "WordShift":
        return """The algorithm splits the **Plaintext** into words based on spaces. Each word is then individually encrypted using the Caesar cipher, resulting in the **Ciphertext**."""
    else:
        raise ValueError(f"Invalid cipher type: {cipher_type}")
