from setuptools import setup, find_packages

setup(
    name="medai",
    version="1.0.0",
    description="Medical Report AI Agent with BiomedBERT, FAISS RAG, and Gemini Agent",
    author="Vansh",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
    ],
)
