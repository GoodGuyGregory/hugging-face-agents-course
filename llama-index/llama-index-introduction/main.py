from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
import colorama
from dotenv import load_dotenv



def main():
    
    
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    
    print(colorama.Fore.CYAN + "Introduction Section of the LlamaHub: ")
    print(colorama.Fore.WHITE + "---------------------------------")
    
    # create the LLM object with the inference object 
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.7,
        max_tokens=100,
        token=hf_token,
        provider="auto"
    )

    print(colorama.Fore.YELLOW + f"Querying the LLM Model: {llm.model_name}")
    
    print(colorama.Fore.WHITE + "---------------------------------")
    
    prompt = input(colorama.Fore.YELLOW + "Enter your Prompt: ")
    print(colorama.Fore.WHITE + "---------------------------------")
    
    response = llm.complete(prompt)
    
    print(colorama.Fore.GREEN + f"🤖: {response}")
    print(colorama.Fore.WHITE + "---------------------------------")

if __name__ == "__main__":
    main()
