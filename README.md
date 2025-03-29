# PDF Chat Application

A Streamlit-based Q&A application that allows users to upload PDF documents and interact with them through natural language queries. The application uses LangChain and OpenAI embeddings to create a vector database from the PDFs, enabling semantic search and contextual answers.

## Features

- Upload multiple PDF files to build a vector database
- Ask questions about the uploaded documents with real-time answers
- PDF validation (format check, minimum size)
- Progress tracking during file processing
- Clear error messages and user feedback
- Environment variable configuration for OpenAI API key

## Directory Structure

```
.
├── .env.template       # Environment variable template
├── .gitignore
├── PDFChat.py          # Main application code
├── README.md
└── requirements.txt    # Python dependencies
```

## Prerequisites

- Python 3.11+
- OpenAI account (for API key)
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set OpenAI API key as an environment variable:
   ```bash
   echo "OPENAI_API_KEY=your-api-key" > .env
   ```

3. (Optional) Configure vector store path in `.env`:
   ```bash
   echo "VECTORES_PATH=/custom/path" >> .env
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run PDFChat.py
   ```

2. Upload PDF files via the web interface
3. Click "Build Vector Store" to process documents
4. Ask questions in the chat interface to get answers based on the documents

## Contributing

To add new features:
1. Create a feature branch
2. Implement changes following PEP8 standards
3. Add tests if applicable
4. Submit a pull request

## License

MIT License - feel free to modify and distribute
