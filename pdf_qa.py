from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔑 Pobierz klucz z .env
openai_key = os.getenv("OPENAI_API_KEY")

# 📄 Ścieżka do pliku PDF
PDF_PATH = "1.pdf"

# Wczytaj PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# Podziel dokument
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = splitter.split_documents(pages)

# Przygotuj AI
prompt = PromptTemplate.from_template(
    "Odpowiedz na pytanie na podstawie dokumentu: {context}\n\nPytanie: {question}"
)
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key)
output_parser = StrOutputParser()

# Połącz prompt → model → parser
chain = prompt | llm | output_parser

# Interaktywna sesja
print("✅ Wczytano dokument. Zadaj pytanie lub wpisz 'exit', aby zakończyć.")

while True:
    question = input("❓ Twoje pytanie: ")
    if question.lower() in ["exit", "quit", "q"]:
        break
    response = chain.invoke({"context": documents, "question": question})
    print("🧠 Odpowiedź AI:", response)