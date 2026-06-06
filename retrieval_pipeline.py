from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

query = input("Enter the query: ")


query_with_prefix = f"query: {query}"

retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query_with_prefix)

print(f"\nUser query: {query}")
print("\n--- Context ---")

context = ""
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    context += f"- {doc.page_content}\n"

combined_input = f"""
You are a helpful assistant.

Answer the question ONLY using the provided context.
If the answer is not in the context, say:
"I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke(combined_input)

print("\n--- Answer ---")
print(response)