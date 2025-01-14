# Document Assistant using RAG

## Description
This project implements a **Retrieval-Augmented Generation (RAG)** system that can retrieve and generate answers based on information from a specific, large document (e.g., a technical manual, legal document, or extensive report). The system uses a **retriever** to fetch relevant passages from the document and a **generator** (based on the T5 model) to generate coherent and accurate answers to user queries.

---

## Requirements
To run this program, you need the following:
1. **Python 3.11** or later.
2. **Docker** (optional, for containerization).
3. The following Python libraries:
   - `transformers` (for the T5 model).
   - `sentence-transformers` (for the retriever).
   - `scikit-learn` (for cosine similarity).
   - `numpy` (for numerical operations).
   - `re` (for text preprocessing).

You can install the required libraries using:
```bash
pip install transformers sentence-transformers scikit-learn numpy
```

---

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/amenallah01/document-assistant.git
cd document-assistant
```

### 2. Set Up the Environment
#### Using Docker (Recommended)
1. Build the Docker image:
   ```bash
   docker build -t document-assistant .
   ```
2. Run the Docker container:
   ```bash
   docker run -it document-assistant
   ```

#### Without Docker
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Program

### 1. Prepare Your Document
Place your document (e.g., `sample_doc.txt`) in the `data` directory. The document should be in plain text format.

### 2. Run the Program
You can execute the program using the following methods:

#### Command Line
1. Navigate to the `src` directory:
   ```bash
   cd src
   ```
2. Run the `main.py` script:
   ```bash
   python main.py
   ```

#### Docker
If you built the Docker image, you can run the program inside the container:
```bash
docker run -it document-assistant
```

---

### Example Usage
#### Input Document (`data/sample_doc.txt`):
```
The warranty period for Product X is 12 months from the date of purchase. Product X must be maintained according to the manual for the warranty to remain valid. If you encounter issues with Product X, contact customer support.
```

#### Query:
```
What is the warranty period for Product X?
```

#### Output:
```
Query: What is the warranty period for Product X?
Context: The warranty period for Product X is 12 months from the date of purchase. Product X must be maintained according to the manual for the warranty to remain valid. If you encounter issues with Product X, contact customer support.
Generated Answer: The warranty period for Product X is 12 months from the date of purchase.
```

---

## File Structure
```
document-assistant/
├── data/                   # Directory for input documents
│   └── sample_doc.txt      # Example document
├── src/                    # Source code
│   ├── main.py             # Main script to run the program
│   ├── retriever.py        # Retriever module
│   └── generator.py        # Generator module
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Customization
- **Model**: You can switch to a different T5 model (e.g., `t5-base` or `t5-large`) by modifying the `model_name` parameter in the `Generator` class.
- **Retriever**: You can customize the retriever by changing the chunk size or using a different embedding model.

---

## Troubleshooting
1. **Invalid API Token**: If you encounter token-related errors, ensure your Hugging Face API token is valid and has the correct permissions.
2. **Model Not Found**: If the T5 model fails to load, ensure you have an active internet connection during the first run to download the model.
3. **Docker Issues**: If Docker fails to build or run, ensure Docker is installed and running on your system.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For further assistance, feel free to open an issue on the repository or contact the maintainers.

---

This README provides a clear and concise guide for users to set up and run your Document Assistant using RAG. Let me know if you need further adjustments!