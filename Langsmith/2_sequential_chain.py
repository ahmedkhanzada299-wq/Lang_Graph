from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM APP'

load_dotenv()

Prompt1 = PromptTemplate(
    template="Generate a detailed report on the following topic: {topic}",
    input_variables=['topic']
)


Prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model="gpt-4o-mini" , temperature=0.7)
model2 = ChatOpenAI(model="gpt-4o" , temperature=0.5)


parser = StrOutputParser()

chain = Prompt1 | model1 | parser| Prompt2 | model2 | parser

config = {
    'run_name': 'Sequential App',
    'tag': ['sequential-chain', 'demo' , 'report-generation' , 'summary-generation'],
    'metadata': {'model1': 'gpt-4o-mini', 'model2': 'gpt-4o' ,'model1_temperature': 0.7, 'model2_temperature': 0.5}
}

response = chain.invoke({"topic": "The impact of climate change on global agriculture."} , config=config)
print(response)

