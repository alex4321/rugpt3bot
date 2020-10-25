from dataclasses import dataclass
from typing import List, Tuple
import re
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer


ALLOW_REPEATS = 'qwertyuiopasdfghjklzxcvbnmйцукенгшщзхъфывапролджэячсмитьбю1234567890' + \
                'QWERTYUIOPASDFGHJKLZXCVBNMЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ'


@dataclass
class Phrase:
    phrase: str
    length: int


@dataclass
class RobotState:
    name: str
    seed: Phrase
    context: List[Phrase]


class Robot:
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
                 query_formatter: str,
                 generator_parameters: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.query_formatter = query_formatter
        self.generator_parameters = generator_parameters

    def _clean_input_text(self, input_text: str) -> str:
        text = input_text + ' '
        while True:
            text_cleaned = []
            for i in range(len(text) - 1):
                if (text[i] in ALLOW_REPEATS) or (text[i] != text[i + 1]):
                    text_cleaned.append(text[i])
            new_text = ''.join(text_cleaned)
            if text.strip() == new_text.strip():
                break
            text = new_text.strip()
        return text.strip()

    def get_phrase_length(self, phrase: str) -> int:
        return len(self.tokenizer.encode(phrase, return_tensors=None))

    def _preprocess_input(self, phrase: str, state: RobotState) -> Tuple[str, RobotState]:
        phrase = self.query_formatter.format(phrase)
        state.context.append(Phrase(phrase=phrase, length=self.get_phrase_length(phrase)))

        keep_last_n_context = 0
        length = state.seed.length
        for i, phrase in enumerate(state.context[::-1]):
            length += phrase.length
            if length < self.model.config.n_ctx:
                keep_last_n_context = i + 1
            else:
                break

        state.context = state.context[-keep_last_n_context:]

        input_texts = [state.seed.phrase] + ["- " + phrase.phrase for phrase in state.context]
        input_text = self._clean_input_text('\n'.join(input_texts) + ' ')
        input_text += "\nВы ответили:\n-"

        return input_text, state

    def _generate(self, input_text: str):
        input_ids = self.tokenizer.encode_plus(input_text, return_tensors='pt')["input_ids"].to(self.model.device)
        output = self.model.generate(input_ids,
                                     max_length=input_ids.shape[1] + 200,
                                     **self.generator_parameters)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        output_text = output_text.replace(input_text.strip(), "").strip()

        return output_text

    def _cut_phrase(self, text: str) -> str:
        text = text.strip("!?. \n")
        for delimeter in "!?.":
            if delimeter in text:
                text = text.split(delimeter)[0] + delimeter
        text = text.strip()
        for delimeter in "!?.":
            if delimeter in text:
                text = text.split(delimeter)[0] + delimeter
                break
        return text

    def _cut_phrases(self, text: str, max_count: int) -> str:
        phrases = []
        count = round(1 + random.random() * (max_count - 1))
        for i in range(count):
            p = self._cut_phrase(text)
            text = text.replace(p, '')
            phrases.append(p)
        return ' '.join(phrases)

    def _clean_output(self, output_text: str) -> str:
        output_text = self._cut_phrases(output_text, 3)
        output_text = re.sub("(-\s*(говор|сказ|спрос|воскл).*-)", '', output_text)

        return output_text

    def answer(self, phrase: str, state: RobotState) -> Tuple[str, RobotState]:
        input_text, new_state = self._preprocess_input(phrase, state)
        output_text = self._generate(input_text)
        output_text = self._clean_output(output_text)
        output_text_length = self.get_phrase_length(output_text)
        new_state.context.append(Phrase(phrase=output_text, length=output_text_length))

        return output_text, new_state

