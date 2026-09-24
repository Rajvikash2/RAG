from google import genai
import os


client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


def generate_answer(context, question):

    prompt = f"""
You are a helpful assistant answering questions about a PDF.

Answer the question using only the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text