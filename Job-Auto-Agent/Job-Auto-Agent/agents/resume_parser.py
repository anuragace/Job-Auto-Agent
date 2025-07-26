import fitz  # PyMuPDF
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a PDF resume."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def load_resume_parser(llm):
    with open("prompts/resume_extract.txt", "r") as f:
        template = f.read()

    prompt = PromptTemplate(input_variables=["resume_text"], template=template)
    return prompt | llm

def parse_resume(resume_text, llm):
    chain = load_resume_parser(llm)
    return chain.invoke({"resume_text": resume_text})
