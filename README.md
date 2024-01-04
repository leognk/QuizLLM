# QuizLLM

QuizLLM is a simple Python program that allows you to play a quiz in the command line with multiple-choice questions generated by an LLM (Large Language Model) running locally, generating questions on any topic of your choice.

## Requirements

The LLM used is OpenHermes 2.5 - Mistral 7B, a state of the art open-source LLM (as of 2024/01/04) with weights downloaded from Hugging Face:
* ~14 GB of disk storage
* ~16 GB of RAM (tested on an RTX 3090)
  
Although not tested, other LLMs would probably work as well.

## How to play

In the command line:
* Run: `python3 main.py`
* Enter a topic of your choice like "biology", "Harry Potter", or more original topics like "pi", "uncomfortable situations", "fear of dogs", or "anything". The possibilities are endless.
* Specify the number of questions you want.
* Play.

## Limitations

* As LLMs hallucinate, it is common that the questions and their answers are wrong.
* On rare occasions, the quiz generation might fail to follow the template.

## How it works

* The program prompts the LLM as follows: "Make a quiz with {n} questions about {topic}, each with 4 choices and the answer."
* The LLM generates the content using a structured template, enabling easy extraction of each entry:
  * Questions: "1. {question}\n", "2. {question}\n", etc.
  * Choices: "a) {choice}\n", "b) {choice}\n", etc.
  * Answer: "Answer: (a|b|c|d)".
* The quiz is then played by interacting with the user.