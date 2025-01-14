from huggingface_hub import InferenceClient

class Generator:
    def __init__(self, model_name="gpt2", api_token=None, debug=False):
        """
        Initialize the generator using Hugging Face's InferenceClient.
        :param model_name: Name of the model to use.
        :param api_token: Your Hugging Face API token.
        :param debug: Enable debug logging.
        """
        self.client = InferenceClient(model=model_name, token=api_token)
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
        Generate an answer using the Hugging Face Inference API.
        :param context: Context text.
        :param question: User's question.
        :return: Generated answer.
        """
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        self.log(f"Input text: {input_text}")

        try:
            # Pass the input text directly to text_generation
            response = self.client.text_generation(input_text)
            self.log(f"API response: {response}")
            return response  # Return the generated text directly
        except Exception as e:
            self.log(f"Error during API call: {e}")
            return f"Error: {e}"