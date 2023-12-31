import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig


def idx2letter(i):
    return chr(ord('a') + i)

def letter2idx(c):
    return ord(c) - ord('a')


class QuizGenerator:
    
    cache_dir = "models"
    model_id = "teknium/OpenHermes-2.5-Mistral-7B"

    def __init__(self, device):
        transformers.logging.set_verbosity_error()

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, torch_dtype=torch.float16, attn_implementation="flash_attention_2",
        ).to(self.device)
        
        self.newline_token = self.encode('\n')[-1]

        self.config = GenerationConfig.from_model_config(self.model.config)
        self.config.pad_token_id = self.tokenizer.eos_token_id
        self.config.eos_token_id = [self.tokenizer.eos_token_id, self.newline_token]
        self.config.do_sample = True
        self.config.temperature = 0.95
    
    def encode(self, text, return_tensor=None):
        """Tokenize text."""
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors=return_tensor).input_ids
        return tokens.to(self.device) if return_tensor else tokens

    def _generate_entry(self, past_tokens: torch.Tensor, head: str, max_tokens: int, force_tokens: list[int] = None) -> tuple[torch.Tensor, str]:
        """Return updated past_tokens and the generated entry."""

        self.config.max_new_tokens = max_tokens
        if force_tokens is not None:
            self.config.force_words_ids = force_tokens
            self.config.num_beams = 2
        else:
            self.config.force_words_ids = None
            self.config.num_beams = 1
        
        in_tokens = torch.cat([past_tokens, self.encode(head, "pt")], dim=1)
        out_tokens = self.model.generate(in_tokens, generation_config=self.config)
        entry = self.tokenizer.decode(out_tokens[0, in_tokens.shape[1]:])
        return out_tokens, entry
    
    def __call__(self, topic: str, n_questions: int, n_choices: int) -> list[dict]:
        """Generate a quiz, ie. a list of questions with their options and answer."""

        system_prompt = "You are an assistant."
        prompt = f"Make a quiz with {n_questions} questions about {topic}, each with {n_choices} choices and the answer."
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        past_tokens = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
        choice_ids = [self.encode(idx2letter(i)) for i in range(n_choices)]
        for x in choice_ids: assert len(x) == 1

        # Generate the quiz.
        quiz = []
        for i in range(n_questions):
            # Generate question.
            blank = "\n\n" if i != 0 else ""
            past_tokens, question = self._generate_entry(past_tokens, head=blank + f"{i + 1}.", max_tokens=100)
            question = question.strip()

            # Generate choices.
            choices = []
            for j in range(n_choices):
                past_tokens, choice = self._generate_entry(past_tokens, head=f"{idx2letter(j)})", max_tokens=100)
                choices.append(choice.strip())

            # Generate answer.
            past_tokens, answer = self._generate_entry(past_tokens, head="Answer:", max_tokens=1, force_tokens=choice_ids)
            answer = letter2idx(answer.strip())
            quiz.append({"question": question, "choices": choices, "answer": answer})
        return quiz