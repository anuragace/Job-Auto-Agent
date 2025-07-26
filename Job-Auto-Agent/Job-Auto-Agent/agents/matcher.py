from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

def load_matcher(llm):
    with open("prompts/match_resume_jd.txt", "r") as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["job_description", "resume"],
        template=template
    )
    return prompt | llm

def match_resume_to_jd(parsed_resume, parsed_jd, llm):
    chain = load_matcher(llm)
    return chain.invoke({
        "job_description": str(parsed_jd),
        "resume": str(parsed_resume)
    })
