import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

def test_models():
    print("--- 1. Testing Google Gemini 3.5 Flash ---")
    try:
        gemini = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.0
        )
        gemini_response = gemini.invoke("Respond with exactly: 'Gemini 3.5 Flash is ready for business.'")
        print(f"Success! Response: {gemini_response.content[0]['text']}\n")
    except Exception as e:
        print(f"Error testing Gemini: {e}\n")

    print("--- 2. Testing Groq (gpt-oss-120b) ---")
    try:
        groq = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.0
        )
        groq_response = groq.invoke("Respond with exactly: 'Groq inference engine is online.'")
        print(f"Success! Response: {groq_response.content.strip()}\n")
    except Exception as e:
        print(f"Error testing Groq: {e}\n")

if __name__ == "__main__":
    test_models()