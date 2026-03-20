from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate

llm = ChatOpenAI(model="gpt-4o")

# 1. Create and/or select your dataset
client = Client()
dataset_name = "polly"


# 2. Define an evaluator
def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    return outputs == reference_outputs

dataset = client.create_dataset(
    dataset_name=dataset_name,
)
