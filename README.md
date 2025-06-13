# 🔐 CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Research](https://img.shields.io/badge/field-Cryptography%20%26%20AI-yellow)

🔗 **Resources:**
- 🤗 [Hugging Face Dataset](https://huggingface.co/datasets/yu0226/CipherBank)
- 📜 [arXiv Paper](https://arxiv.org/abs/2504.19093)
- 🚀 [Project page](https://cipherbankeva.github.io/)

> 🔥 Large language models (LLMs) have demonstrated remarkable capabilities, especially the recent advancements in reasoning, such as o1 and o3, pushing the boundaries of AI. Despite these impressive achievements in mathematics and coding, the reasoning abilities of LLMs in domains requiring cryptographic expertise remain underexplored.

## 🔥 News
2025.05:  🎉🎉 Congratulations: CipherBank was accepted by ACL-2025 finding conference.

## 📜 Abstract

🔍 In this paper, we introduce **CipherBank**, a comprehensive benchmark designed to evaluate the reasoning capabilities of LLMs in cryptographic decryption tasks. CipherBank comprises **2,358** meticulously crafted problems, covering **262 unique plaintexts** across **5 domains** and **14 subdomains**, with a focus on privacy-sensitive and real-world scenarios that necessitate encryption.

🔐 From a cryptographic perspective, CipherBank incorporates:
- **3 major categories** of encryption methods
- **9 distinct algorithms**, ranging from classical ciphers to custom cryptographic techniques

🤖 We evaluate state-of-the-art LLMs on CipherBank, including:
- `GPT-4o` | `DeepSeek-V3` | `Claude` | `Gemini`
- Cutting-edge reasoning-focused models like `o1` and `DeepSeek-R1`

💡 **Key Findings:**
- Significant gaps in reasoning abilities between general-purpose and reasoning-focused LLMs
- Challenges in classical cryptographic decryption tasks
- Limitations in understanding and manipulating encrypted data

## 📂 Data Introduction

| File | Description |
|------|-------------|
| `data/plaintext.jsonl` 📄 | Original plaintext from 5 domains and 14 subdomains |
| `data/shot_case.jsonl` 🎯 | 3 case examples for few-shot testing |
| `data/test.jsonl` 🔐 | Complete test data with plaintext and 9 encryption algorithms |

## 🧪 Test Introduction

### 🔒 Encryption
```bash
python cipher/encryption.py --input_file ../data/plaintext.jsonl --output_file ../data/test.jsonl --mode cipher
```

### 🔓 Decryption (Test Reversibility)
```bash
python cipher/encryption.py --input_file ../data/test.jsonl --mode decrypt
```

## 🤖 Test Your Model

### 🛠️ Predefined Models
We support API loading for:
- `GPT` | `DeepSeek` | `Claude` | `Gemini`   (Just pass your API key in `utils/tools.py`)

### 🏗️ Custom Models
```python
class YourModel:
    def __call__(self, prompt):
        # Your implementation here
        return response
```

### 🏃‍♂️ Run Tests
```bash
# Basic test
bash run.sh --model model_name --shot_number 3

# With detailed prompts
bash run.sh --model model_name --shot_number 3 --is_hint True

# Specific algorithm test (e.g., Rot13)
python test.py --cipher_type Rot13 --model model_name
```

## 📜 Citation
If you find **CipherBank** useful for your your research and applications, please kindly cite using this BibTeX:
```bibtex
@misc{li2025cipherbankexploringboundaryllm,
      title={CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges}, 
      author={Yu Li and Qizhi Pei and Mengyuan Sun and Honglin Lin and Chenlin Ming and Xin Gao and Jiang Wu and Conghui He and Lijun Wu},
      year={2025},
      eprint={2504.19093},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2504.19093}, 
}
```

## 🤝 Contributing
PRs welcome! Please open an issue first to discuss changes.
