import datasets
from llama_index.core.schema import Document
import colorama

def load_guest_dataset(confirm_load=False):
    
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    
    docs = [
        Document(
            text="\n".join([
                f"Name: {guest_dataset['name'][i]}",
                f"Relation: {guest_dataset['relation'][i]}",
                f"Description: {guest_dataset['description'][i]}",
                f"Email: {guest_dataset['email'][i]}",
            ]),
            metadata={"name": guest_dataset['name'][i]}
        ) for i in range(len(guest_dataset))
    ]
    
    if confirm_load:
        
        for guest in docs:
            print(colorama.Fore.GREEN + f"✅ Added Guest: {guest.metadata}")