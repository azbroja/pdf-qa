from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔑 Load OpenAI API key from .env
openai_key = os.getenv("OPENAI_API_KEY")

# 📄 Path to the PDF file
PDF_PATH = "your_file.pdf"

# 📚 Load the PDF document
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# ✂️ Split the document into smaller chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = splitter.split_documents(pages)

# 🧠 Prepare prompt, model and parser
prompt = PromptTemplate.from_template(
    "Answer the question based on the following document: {context}\n\nQuestion: {question}"
)
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key)
output_parser = StrOutputParser()

# 🔗 Create the chain
chain = prompt | llm | output_parser

# 🧪 Interactive loop
print("✅ Document loaded. Ask a question or type 'exit' to quit.")

while True:
    question = input("❓ Your question: ")
    if question.lower() in ["exit", "quit", "q"]:
        break
    response = chain.invoke({"context": documents, "question": question})
    print("🧠 AI Answer:", response)