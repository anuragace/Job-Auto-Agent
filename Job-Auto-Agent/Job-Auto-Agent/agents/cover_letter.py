from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

def load_cover_letter_chain(llm):
    with open("prompts/cover_letter.txt", "r") as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["job_description", "resume"],
        template=template,
    )
    return prompt | llm

def generate_cover_letter(job_description: str, resume: str, llm):
    chain = load_cover_letter_chain(llm)
    return chain.invoke({
        "job_description": job_description,
        "resume": resume
    })
