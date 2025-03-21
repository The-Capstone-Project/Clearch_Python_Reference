# Clearch: CLI System Information Assistant



## Overview

Clearch is a specialized virtual assistant for terminal environments that provides detailed system information. It leverages natural language processing to understand user queries and responds with precise system data extracted from your computer.

With Clearch, you can:
- Query hardware and software information
- Troubleshoot system issues
- Get technical guidance for your specific system
- Obtain step-by-step instructions compatible with your configuration

## The thing in action

![image](https://github.com/user-attachments/assets/713f16ae-f0b6-4bd3-adc1-34735c92838b)
![image](https://github.com/user-attachments/assets/b04739ba-236b-4436-82f8-15c3a0e05e05)


## Features

- **Natural Language Understanding**: Ask questions in plain English about your system
- **Context-Aware Responses**: Provides answers based on your actual system specifications
- **Technical Accuracy**: Extracts precise information about memory, disk space, hardware, etc.
- **Formatting for Readability**: Presents system data in a clean, formatted style
- **Step-by-Step Solutions**: Offers troubleshooting guidance tailored to your system

## Architecture

Clearch uses a combination of technologies to deliver accurate responses:

1. **Vector Database**: System information is stored in a Chroma vector database
2. **Embedding Model**: Uses Hugging Face's sentence-transformers for semantic search
3. **LLM Integration**: Leverages Groq's LLMs for intelligent response generation
4. **RAG (Retrieval Augmented Generation)**: Combines retrieved system context with powerful LLM capabilities

## Requirements

- Python 3.8+
- pip package manager
- Groq API key ([sign up here](https://console.groq.com/))

## Installation

1. Clone the repository or download the project files:

```bash
git clone https://github.com/yourusername/clearch.git
cd clearch
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:

```bash
cp .env.example .env
```

4. Edit the `.env` file and add your Groq API key:
