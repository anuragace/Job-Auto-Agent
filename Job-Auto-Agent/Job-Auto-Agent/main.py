import os 
from langchain_openai import ChatOpenAI
from agents.jd_parser import parse_jd
from agents.resume_parser import extract_text_from_pdf, parse_resume
from langchain_core.prompts import PromptTemplate
from agents.cover_letter import generate_cover_letter

job_description = """
We are seeking a talented and driven AI Engineer to join our cross-functional AI innovation team within a leading pharmaceutical organization. You will work on designing, developing, and deploying GenAI-driven solutions, retrieval-augmented generation (RAG) systems, and scalable ML-powered applications across cloud-native environments to support drug development, regulatory, manufacturing, and commercial operations. The ideal candidate combines a strong academic foundation in machine learning with hands-on experience building real-world AI systems and full-stack applications that comply with the high standards of a regulated industry.

Essential Functions

Design and implement GenAI applications including LLM-integrated tools, RAG systems, and intelligent chatbots tailored for pharmaceutical functions such as R& D knowledge mining, regulatory intelligence, and medical information services.
Build and optimize data ingestion, transformation, and semantic search pipelines using domain-specific embedding models (e.g., BioBERT, Ada-002).
Develop and containerize AI services using FastAPI, Docker, and Kubernetes, deployed on AWS or GCP, ensuring adherence to data privacy and compliance standards (e.g., HIPAA, GxP).
Collaborate with product, data, and platform engineering teams to integrate AI features into validated systems supporting clinical, quality, and manufacturing workflows.
Contribute to end-to-end MLOps: model experimentation, deployment pipelines, audit readiness, monitoring, and performance tuning.
Engineer full-stack applications (React + REST/GraphQL backends) with role-based access controls for internal and external scientific and business users.
Support internal research and POCs on advanced topics like LLM reasoning, adverse event detection, pharmacovigilance, and digital biomarker analysis.

Requirements

Education:

Bachelor’s Degree (BA/BS) BS in Computer Science - Required
Master Degree (MS/MA) M.S. in Computer Science (Machine Learning track) or equivalent experience - Preferred

Experience

2 years or more in Experience in Python, SQL, and ML libraries like PyTorch, TensorFlow, scikit-learn.
2 years or more in Demonstrated experience with GenAI frameworks such as LangChain, Hugging Face, OpenAI APIs.
2 years or more in Exposure to healthcare or life sciences data, regulated environments, or ontologies like UMLS or SNOMED
2 years or more in Experience working in R& D or regulatory settings within the pharmaceutical or biotech industry
2 years or more in Experience with semantic search, LaTeX OCR pipelines, or financial/news sentiment models

Skills

A "builder's mindset" with a strong passion for translating AI research into useful tools in support of patient impact. - Intermediate
Familiarity with MLOps workflows, DevOps tools (GitHub Actions, CI/CD), and cloud platforms (AWS, GCP) - Intermediate
Experience with semantic search, LaTeX OCR pipelines, or financial/news sentiment models - Beginner

Specialized Knowledge

Proficiency in Python, SQL, and ML libraries like PyTorch, TensorFlow, scikit-learn.
Demonstrated experience with GenAI frameworks such as LangChain, Hugging Face, OpenAI APIs
Experience developing cloud-native microservices using FastAPI, Flask, and deploying via Docker/K8s
Familiarity with MLOps workflows, DevOps tools (GitHub Actions, CI/CD), and cloud platforms (AWS, GCP)
.
"""
os.environ["OPENAI_API_KEY"] = os.environ.get("GROQ_API_KEY")
# 🌐 Connect to model (e.g., Groq, Together, OpenRouter)
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",  
    model="llama3-70b-8192",
)

# ✅ Run the JD parser agent
result = parse_jd(job_description, llm)
print(result)

# 📄 Path to your local resume file
resume_path = "resumes/Anurag_Kalapala_ML Engineer.pdf"  
# ✂️ Extract raw text from PDF
resume_text = extract_text_from_pdf(resume_path)

# 🧠 Run the resume parser agent
parsed_resume = parse_resume(resume_text, llm)

print(parsed_resume)

# 🧠 Load matching prompt
with open("prompts/match_resume_jd.txt", "r") as f:
    match_template = f.read()

match_prompt = PromptTemplate(
    input_variables=["job_description", "resume"],
    template=match_template
)

match_chain = match_prompt | llm

# 🤝 Run resume-JD matching
match_result = match_chain.invoke({
    "job_description": result,      # Parsed JD dict
    "resume": parsed_resume         # Parsed resume dict
})

# 📤 Print final match insights
print("\n\n📊 MATCH REPORT:\n")
print(match_result)

# 📨 Step 5: Generate the customized cover letter
cover_letter = generate_cover_letter(job_description, resume_text, llm)

print("\n📄 Cover Letter:\n")
print(cover_letter)

