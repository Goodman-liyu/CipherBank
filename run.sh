#!/bin/bash

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [--model model_name] [--shot_number number] [--is_hint provide_hint]"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --shot_number)
            SHOT_NUMBER="$2" 
            shift 2
            ;;
        --is_hint)
            IS_HINT="$2" 
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set default values
MODEL=${MODEL:-"gpt-4o-2024-08-06"}
SHOT_NUMBER=${SHOT_NUMBER:-3}
IS_HINT=${IS_HINT:-"False"}
if [ "$IS_HINT" = "True" ]; then 
    python test.py --cipher_type rot13 --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
    python test.py --cipher_type atbash --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
    python test.py --cipher_type vigenere --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
    python test.py --cipher_type polybius --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
    python test.py --cipher_type reverse --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
    python test.py --cipher_type swap_pairs --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &     
    python test.py --cipher_type lsb --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
    python test.py --cipher_type openai --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
    python test.py --cipher_type word_shift --model "$MODEL" --shot_number "$SHOT_NUMBER" --is_hint &
else
    python test.py --cipher_type rot13 --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type atbash --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type vigenere --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type polybius --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type reverse --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type swap_pairs --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type lsb --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type openai --model "$MODEL" --shot_number "$SHOT_NUMBER" &
    python test.py --cipher_type word_shift --model "$MODEL" --shot_number "$SHOT_NUMBER" &
fi

# Wait for all background processes to complete
wait