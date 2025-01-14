from retriever import Retriever
from generator import Generator

if __name__ == "__main__":
    # Replace this with your actual Hugging Face API token
    HF_API_TOKEN = "your_new_huggingface_api_token"

    # Initialize the retriever and generator
    retriever = Retriever(chunk_size=50)
    generator = Generator(model_name="gpt2", api_token=HF_API_TOKEN, debug=True)

    # Load the document
    document_path = "data/sample_doc.txt"
    retriever.load_document(document_path)

    # Query the system
    query = "What is the warranty period for Product X?"
    top_k = 2
    retrieved = retriever.retrieve(query, top_k=top_k)

    # Combine retrieved contexts
    context = " ".join([text for text, score in retrieved])
    
    # Generate an answer
    answer = generator.generate(context, query)

    # Print results
    print("Query:", query)
    print("Context:", context)
    print("Generated Answer:", answer)