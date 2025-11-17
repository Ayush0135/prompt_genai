import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
)

# Streamlit UI
st.header('Reasearch Tool')
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(
    template="""You are an expert AI research explainer.

Explain the research paper titled:
"{paper_input}"

Explanation Style: {style_input}
Length: {length_input}

Requirements:
- Maintain accuracy and depth.
- Follow the selected style.
- Use clear structure and easy readability.
- Beginner-friendly → simplify
- Technical → include architecture details
- Code-oriented → include code/pseudocode
- Mathematical → include equations

Output Format:
1. Introduction
2. Key Contributions
3. Method/Architecture
4. Important Insights
5. Why It Matters

Start explaining now.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)