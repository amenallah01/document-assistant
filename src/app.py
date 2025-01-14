########################################################################
####               Document Assistant Using RAG                     ####
####                     KRISSAAN AMEN ALLAH                        ####        
########################################################################
####                             UI                                 #### 
########################################################################

import gradio as gr
from retriever import Retriever
from generator import Generator

# Initialize the retriever and generator
retriever = Retriever(chunk_size=50)
generator = Generator(model_name="t5-small", debug=True)

# Load the document
document_path = "data/sample_doc.txt"
retriever.load_document(document_path)

def answer_question(question):
    # Retrieve relevant context
    retrieved = retriever.retrieve(question, top_k=2)
    context = " ".join([text for text, score in retrieved])
    
    # Generate an answer
    answer = generator.generate(context, question)
    return answer

# Create a Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="Document Assistant",
    description="Ask a question about the document."
)

# Launch the interface
iface.launch()