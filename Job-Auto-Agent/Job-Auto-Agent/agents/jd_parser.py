from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

def load_jd_parser(llm):
    with open("prompts/jd_extract.txt", "r") as f:
        template = f.read()

    prompt = PromptTemplate(input_variables=["job_description"], template=template)
    return prompt | llm  # This creates a RunnableSequence

def parse_jd(jd_text, llm):
    chain = load_jd_parser(llm)
    return chain.invoke({"job_description": jd_text})  # instead of .run()