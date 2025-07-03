import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Loading environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise EnvironmentError("❌ GROQ_API_KEY is not set. Please add it in Streamlit secrets or environment variables.")

# Set the environment variable explicitly if needed by downstream libraries
os.environ["GROQ_API_KEY"] = api_key


# Initializing the LLM
llm=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
)

def generate_report(disease: str, confidence: float, patient_info: str) -> str:
    prompt = f"""
        You are a professional radiologist. Based on the following input, generate a formal and accurate radiology report.

        Patient Information:
        {patient_info}

        CNN Model Prediction:
        Disease: {disease}
        Confidence: {confidence:.2f}

        Report:

        ! Don't include signature or any personal identifiers.
        """

    # Prompt Template
    try:
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system",f"You are a professional radiologist. Use the {prompt} to generate a detailed AI radiology report. Include sections like Patient Information, Clinical Presentation, Radiological Findings, Additional Findings, Conclusion, Recommendations, and Impression."),
                ("user","Question:{prompt}"),
            ]
        )

        output_parser=StrOutputParser()
        chain=prompt|llm|output_parser
        report = chain.invoke({"prompt": prompt})
        return report.strip()
    except Exception as e:
        return f"Error generating report: {str(e)}"