from openai import OpenAI

def translate_critique(text):
  client = OpenAI()
  prompt =f"""I'll give you a text in English and I need you to translate it as accurately as possible to Spanish.
  Try to keep the overall tone of the text as much as possible. Return only the translation. Here's the text:
  {text}"""
  response = client.responses.create(
      model="gpt-4o-mini",
      input=[{"role":"system", "content":"You're a helpful translation assistant specialized in English-Spanish translation with deep knowledge in both american and latin american culture."},
                {"role":"user", "content": prompt}]
  )
  translation = response.output_text
  return translation
