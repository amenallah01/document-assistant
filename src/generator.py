########################################################################
####               Document Assistant Using RAG                     ####
####                     KRISSAAN AMEN ALLAH                        ####        
########################################################################
####                          Generator                             #### 
########################################################################


from transformers import T5ForConditionalGeneration, T5Tokenizer

class Generator:
    def __init__(self, model_name="t5-small", debug=False):
        """
        Initialize the generator using the T5 model.
        :param model_name: Name of the T5 model to use (e.g., "t5-small", "t5-base", "t5-large").
        :param debug: Enable debug logging.
        """
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.debug = debug

    def log(self, message):
        """
        Log a message if debug mode is enabled.
        :param message: The message to log.
        """
        if self.debug:
            print(f"[DEBUG] {message}")

    def generate(self, context, question):
        """
        Generate an answer using the T5 model.
        :param context: Context text.
        :param question: User's question.
        :return: Generated answer.
        """
        # Format the input for T5
        input_text = f"question: {question} context: {context}"
        self.log(f"Input text: {input_text}")

        try:
            # Tokenize the input
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate the output
            outputs = self.model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
            
            # Decode the output to text
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.log(f"Generated answer: {answer}")
            return answer
        except Exception as e:
            self.log(f"Error during generation: {e}")
            return f"Error: {e}"