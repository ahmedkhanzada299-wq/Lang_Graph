from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate.from_template("{question}")

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"question": "What is the capital of USA?"})

print(response)



