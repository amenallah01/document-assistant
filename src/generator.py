from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name="t5-small"):
        """
        Initialize the generator with a pre-trained model.
        :param model_name: Name of the pre-trained model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, context, question, max_length=100):
        """
        Generate an answer based on the context and question.
        :param context: Retrieved context text.
        :param question: User's question.
        :param max_length: Maximum length of the generated response.
        :return: Generated answer as a string.
        """
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
