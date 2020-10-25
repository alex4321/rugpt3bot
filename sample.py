from transformers import GPT2LMHeadModel, GPT2Tokenizer
from robot import RobotState, Robot, Phrase
import torch


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt3_large_hf")
    model = GPT2LMHeadModel.from_pretrained("gpt3_large_hf")
    if torch.cuda.device_count():
        model = model.cuda().half()
    else:
        model = model.cpu()
    seed_text = "Вы - Саймон Джарретт, родились 16 июля 1988 года, в маленьком канадском городке. " + \
        "Вы любите кино и видеоигры, в 2015 годы вы работаете в книжном магазине Гримуар в Торонто. " + \
        "Обычно вы весьма добродушны, но сейчас вы подавлены из-за недавней трамы головы, полученной " + \
        "при автокатастрофе. Кроме того, в этой же катастрофе погибла ваша коллега - Эшли. " + \
        "Иногда людям кажется, что вы несколько глуповаты - и на то есть причины. " + \
        "Недавно вы учавствовали в следующем диалоге: "
    robot = Robot(model=model,
                  tokenizer=tokenizer,
                  query_formatter="{0} - спросили вас.",
                  generator_parameters={
                      "temperature": 0.5,
                      "top_k": 5,
                      "top_p": 0.95,
                      "repetition_penalty": 4.0,
                      "num_return_sequences": 1,
                      "length_penalty": 1.5,
                      "do_sample": False
                  })
    state = RobotState(
        name="TEST",
        seed=Phrase(phrase=seed_text, length=robot.get_phrase_length(seed_text)),
        context=[]
    )
    while True:
        phrase = input(">>>").strip()
        if phrase == "":
            break
        answer, state = robot.answer(phrase, state)
        print(answer)

