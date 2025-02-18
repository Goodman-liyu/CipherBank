# CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges

This repository includes a python implemenation of `CipherBank`.

> Large language models (LLMs) have demonstrated remarkable capabilities, especially the recent advancements in reasoning, such as o1 and o3, pushing the boundaries of AI. Despite these impressive achievements in mathematics and coding, the reasoning abilities of LLMs in domains requiring cryptographic expertise remain underexplored.In this paper, we introduce CipherBank, a comprehensive benchmark designed to evaluate the reasoning capabilities of LLMs in cryptographic decryption tasks. CipherBank comprises 2,358 meticulously crafted problems, covering 262 unique plaintexts across 5 domains and 14 subdomains, with a focus on privacy-sensitive and real-world scenarios that necessitate encryption. From a cryptographic perspective, CipherBank incorporates 3 major categories of encryption methods, spanning 9 distinct algorithms, ranging from classical ciphers to custom cryptographic techniques.We evaluate state-of-the-art LLMs on CipherBank, e.g., GPT-4o, DeepSeek-V3, and cutting-edge reasoning-focused models such as o1 and DeepSeek-R1. Our results reveal significant gaps in reasoning abilities not only between general-purpose chat LLMs and reasoning-focused LLMs but also in the performance of current reasoning-focused models when applied to classical cryptographic decryption tasks, highlighting the challenges these models face in understanding and manipulating encrypted data. Through detailed analysis and error investigations, we provide several key observations that shed light on the limitations and potential improvement areas for LLMs in cryptographic reasoning.These findings underscore the need for continuous advancements in LLM reasoning capabilities.

## Data Introduction
- `data/plaintext.jsonl` contains the original plaintext from 5 domains and 11 domains.
- `data/shot_case.jsonl` provides 3 case examples used in the few-shot testing.
- `data/test.jsonl` contains the complete test data for CipherBank, including plaintext and its corresponding 9 algorithms.

## Test Introduction:

- You can encrypt `plaintext.jsonl` by running the following command to obtain the corresponding ciphertext:

  ```bash
  python cipher/encryption.py --input_file ../data/plaintext.jsonl --output_file ../data/test.jsonl --mode cipher
  ```

- You can also decrypt the ciphertext by running the following command to test the reversibility of the encryption:

  ```bash
  python cipher/encryption.py --input_file ../data/test.jsonl --mode decrypt
  ```

## Test Your Model:

- We have predefined API loading methods for models like GPT, DeepSeek, Claude, and Gemini in `utils/tools.py`. You only need to pass your own API key to use them directly.

- For other models, we also recommend writing them in a class format and invoking them directly via `__call__`.

- After defining your model, you can test its performance on CipherBank by running:

  ```bash
  bash run.sh --model model_name --shot_number 3
  ```

  Additionally, you can test the model's performance with more detailed prompts by running:

  ```bash
  bash run.sh --model model_name --shot_number 3 --is_hint True
  ```

- You can also test the model's performance on a specific algorithm (e.g., Rot13) by running:

  ```bash
  python test.py --cipher_type Rot13 --model model_name
  ```