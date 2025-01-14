from retriever import Retriever

if __name__ == "__main__":
    # Initialize the retriever
    retriever = Retriever()
    
    # Load the document
    document_path = "data/sample_doc.txt"
    retriever.load_document(document_path)
    
    # Query the retriever
    query = "What is the warranty period for Product X?"
    results = retriever.retrieve(query, top_k=3)
    
    # Print results
    print("Query:", query)
    print("Top Results:")
    for sentence, score in results:
        print(f"- {sentence} (Score: {score:.4f})")
