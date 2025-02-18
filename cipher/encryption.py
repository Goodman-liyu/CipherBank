import json
import base64
from abc import ABC, abstractmethod
from cryptography.fernet import Fernet
import argparse
import os
import math


class CipherAlgorithm(ABC):
    @abstractmethod
    def encrypt(self, text: str) -> bytes:
        pass

    @abstractmethod
    def decrypt(self, encrypted_text: bytes) -> str:
        pass


class CaesarCipher(CipherAlgorithm):
    def __init__(self, shift=3):
        self.shift = shift

    def encrypt(self, text: str) -> bytes:
        result = ""
        for char in text:
            if char.isascii() and char.isalpha():
                ascii_offset = ord("A") if char.isupper() else ord("a")
                result += chr((ord(char) - ascii_offset + self.shift) % 26 + ascii_offset)
            else:
                result += char
        return result

    def decrypt(self, encrypted_text: bytes) -> str:
        text = encrypted_text
        result = ""
        for char in text:
            if char.isascii() and char.isalpha():
                ascii_offset = ord("A") if char.isupper() else ord("a")
                result += chr((ord(char) - ascii_offset - self.shift) % 26 + ascii_offset)
            else:
                result += char
        return result


class AtbashCipher(CipherAlgorithm):
    def encrypt(self, text: str) -> bytes:
        result = ""
        for char in text:
            if char.isascii() and char.isalpha():
                if char.isupper():
                    # Ensure the input character is within the ASCII range of letters
                    if 65 <= ord(char) <= 90:  # ASCII range for A-Z
                        result += chr(90 - (ord(char) - 65))
                    else:
                        result += char
                else:
                    # Ensure the input character is within the ASCII range of letters
                    if 97 <= ord(char) <= 122:  # ASCII range for a-z
                        result += chr(122 - (ord(char) - 97))
                    else:
                        result += char
            else:
                result += char
        return result

    def decrypt(self, encrypted_text: bytes) -> str:
        return self.encrypt(encrypted_text)


class PolybiusCipher(CipherAlgorithm):
    def __init__(self):
        self.matrix = [
            ["A", "B", "C", "D", "E", "F"],
            ["G", "H", "I", "J", "K", "L"],
            ["M", "N", "O", "P", "Q", "R"],
            ["S", "T", "U", "V", "W", "X"],
            ["Y", "Z", "1", "2", "3", "4"],
            ["5", "6", "7", "8", "9", "0"],
        ]
        self.char_to_pos = {}
        for i in range(6):
            for j in range(6):
                self.char_to_pos[self.matrix[i][j]] = f"{i+1}{j+1}"

    def encrypt(self, text: str) -> str:
        result = []
        words = text.split()
        for word in words:
            word_result = []
            for char in word:
                if char.isalpha() and char.isascii():
                    word_result.append(self.char_to_pos.get(char.upper(), char.upper()))
                else:
                    word_result.append(char)
            result.append(" ".join(word_result))
        return "   ".join(result)

    def decrypt(self, encrypted_text: str) -> str:
        text = encrypted_text
        result = []

        words = text.split("   ")
        for word in words:
            numbers = word.split()
            word_result = []

            for num in numbers:
                if len(num) == 2 and num.isdigit():
                    row = int(num[0]) - 1
                    col = int(num[1]) - 1
                    if 0 <= row < 6 and 0 <= col < 6:
                        word_result.append(self.matrix[row][col])
                else:
                    word_result.append(num)

            result.append("".join(word_result))

        return " ".join(result)


class VigenereCipher(CipherAlgorithm):
    def __init__(self, key):
        self.key = key.upper()

    def encrypt(self, text: str) -> bytes:
        result = ""
        key_length = len(self.key)
        key_as_int = [ord(i) - ord("A") for i in self.key]

        for i, char in enumerate(text):
            if char.isascii() and char.isalpha():

                is_upper = char.isupper()

                char_num = ord(char.upper()) - ord("A")

                key_num = key_as_int[i % key_length]

                encrypted_num = (char_num + key_num) % 26
                encrypted_char = chr(encrypted_num + ord("A"))
                result += encrypted_char if is_upper else encrypted_char.lower()
            else:
                result += char
        return result

    def decrypt(self, encrypted_text: bytes) -> str:
        text = encrypted_text
        result = ""
        key_length = len(self.key)
        key_as_int = [ord(i) - ord("A") for i in self.key]

        for i, char in enumerate(text):
            if char.isascii() and char.isalpha():
                is_upper = char.isupper()

                char_num = ord(char.upper()) - ord("A")

                key_num = key_as_int[i % key_length]

                decrypted_num = (char_num - key_num) % 26

                decrypted_char = chr(decrypted_num + ord("A"))
                result += decrypted_char if is_upper else decrypted_char.lower()
            else:
                result += char
        return result


class ReverseCipher(CipherAlgorithm):
    def __init__(self):
        pass

    def encrypt(self, text: str) -> bytes:
        return text[::-1]

    def decrypt(self, encrypted_text: bytes) -> str:
        return encrypted_text[::-1]


class SwapPairsCipher:
    def __init__(self):
        pass

    def encrypt(self, text: str) -> str:
        """
        Encrypts the text by swapping adjacent characters.
        If the length of the text is odd, the last character remains unchanged.
        """
        result = []
        i = 0
        while i < len(text) - 1:
            # Swap adjacent characters
            result.append(text[i + 1])
            result.append(text[i])
            i += 2
        if i < len(text):  # If odd length, add the last character as is
            result.append(text[i])
        return "".join(result)

    def decrypt(self, text: str) -> str:
        """
        Decrypts the text by swapping adjacent characters.
        The process is identical to encryption as swapping is symmetric.
        """
        return self.encrypt(text)  # Symmetric operation


class ParityShiftCipher(CipherAlgorithm):
    def __init__(self):
        pass

    def encrypt(self, text: str) -> bytes:
        result = ""
        for char in text:
            if char.isascii() and char.isalpha():
                ascii_offset = ord("A") if char.isupper() else ord("a")
                shift = 1 if ord(char) % 2 == 0 else -1
                result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
            else:
                result += char
        return result

    def decrypt(self, text: bytes) -> str:
        return self.encrypt(text)


class DualAvgCodeCipher(CipherAlgorithm):
    def __init__(self):
        pass

    def encrypt(self, text: str) -> bytes:
        result = ""
        for char in text:
            if char.isascii() and char.isalpha():
                if char == "a" or char == "z" or char == "A" or char == "Z":
                    result += char * 2
                else:
                    ascii_number = ord(char)
                    left = ascii_number - 1
                    right = ascii_number + 1

                    result += chr(left) + chr(right)
            else:
                result += char
        return result

    def decrypt(self, text: bytes) -> str:
        result = ""
        i = 0
        while i < len(text):
            char = text[i]

            if char.isascii() and char.isalpha():
                left = char
                right = text[i + 1]
                result += chr(int((ord(left) + ord(right)) / 2))
                i += 1
            else:
                result += char
            i += 1
        return result


class WordShiftCipher:
    def __init__(self, shift: int = 3):
        self.shift = shift

    def encrypt(self, text: str) -> str:
        """
        Encrypts the text by treating each word (sequence of non-space characters) as a block
        and performing a left shift within each word.
        """
        words = text.split(" ")  # Split text into words by spaces
        result = []
        for word in words:
            shift = self.shift % len(word) if len(word) > 0 else 0  # Handle cases where word length < shift
            result.append(word[shift:] + word[:shift])
        return " ".join(result)

    def decrypt(self, text: str) -> str:
        """
        Decrypts the text by reversing the left shift applied during encryption.
        """
        words = text.split(" ")  # Split text into words by spaces
        result = []
        for word in words:
            shift = self.shift % len(word) if len(word) > 0 else 0  # Handle cases where word length < shift
            result.append(word[-shift:] + word[:-shift])
        return " ".join(result)


class CipherProcessor:
    def __init__(self, cipher_algorithm: CipherAlgorithm, cipher_name: str):
        self.cipher = cipher_algorithm
        self.cipher_name = cipher_name

    def process_jsonl(self, input_file, mode="cipher", output_file=None):
        if mode == "cipher":
            # If no output file is specified, create a default output file
            if output_file is None:
                output_file = os.path.splitext(input_file)[0] + "_encrypted.jsonl"

            try:
                # If it's the first cipher, create a new file; otherwise, read the existing file
                mode = "r+" if os.path.exists(output_file) else "w+"

                with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, mode, encoding="utf-8") as f_out:

                    # If the file exists, read the existing content
                    existing_data = {}
                    if mode == "r+":
                        f_out.seek(0)
                        lines = f_out.readlines()
                        if lines:
                            existing_data = {i: json.loads(line.strip()) for i, line in enumerate(lines)}
                        f_out.seek(0)

                    # Process the input file
                    for i, line in enumerate(f_in):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Get existing data or create new data
                            data = existing_data.get(i, json.loads(line))

                            if "plaintext" in data:
                                encrypted_content = self.cipher.encrypt(data["plaintext"])
                                data[self.cipher_name] = encrypted_content

                            existing_data[i] = data

                        except json.JSONDecodeError as e:
                            print(f"Warning: The JSON format of the {i+1} line is invalid: {e}")
                            continue

                    # Write all data
                    f_out.seek(0)
                    for data in existing_data.values():
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f_out.truncate()

            except Exception as e:
                print(f"Error processing file: {str(e)}")
        elif mode == "decrypt":
            with open(input_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    data = json.loads(line.strip())
                    decrypted_content = self.cipher.decrypt(data[self.cipher_name])
                    if decrypted_content.lower() != data["plaintext"].lower():
                        print(len(data["plaintext"]))
                        print(len(decrypted_content))
                        print(f"Decryption failed: {data['plaintext']} -> {decrypted_content}")


def main(args):

    caesar_cipher = CaesarCipher(shift=13)
    atbash_cipher = AtbashCipher()
    vigenere_cipher = VigenereCipher(key="ACL")
    polybius_cipher = PolybiusCipher()
    reverse_cipher = ReverseCipher()
    swap_pairs_cipher = SwapPairsCipher()
    lsb_cipher = ParityShiftCipher()
    openai_cipher = DualAvgCodeCipher()
    word_shift_cipher = WordShiftCipher(shift=3)

    processors = [
        (CipherProcessor(caesar_cipher, "Rot13"), "Rot13"),
        (CipherProcessor(atbash_cipher, "Atbash"), "Atbash"),
        (CipherProcessor(polybius_cipher, "Polybius"), "Polybius"),
        (CipherProcessor(vigenere_cipher, "Vigenere"), "Vigenere"),
        (CipherProcessor(reverse_cipher, "Reverse"), "Reverse"),
        (CipherProcessor(swap_pairs_cipher, "SwapPairs"), "SwapPairs"),
        (CipherProcessor(lsb_cipher, "ParityShift"), "ParityShift"),
        (CipherProcessor(openai_cipher, "DualAvgCode"), "DualAvgCode"),
        (CipherProcessor(word_shift_cipher, "WordShift"), "WordShift"),
    ]

    for processor, name in processors:
        print(f"Using {name} for processing...")
        if args.mode == "cipher":
            processor.process_jsonl(args.input_file, mode="cipher", output_file=args.output_file)
        elif args.mode == "decrypt":
            processor.process_jsonl(args.output_file, mode="decrypt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="../data/plaintext.jsonl",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="../data/test.jsonl",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--mode",
        default="cipher",
        type=str,
        help="Processing mode: cipher (encryption), decrypt (decryption)",
    )
    args = parser.parse_args()
    main(args)
