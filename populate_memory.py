import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

store_id = None
stores = client.beta.memory_stores.list()
for s in stores.data:
    if s.name == "mh_memory_store":
        store_id = s.id
        break

if store_id is None:
    print("Memory store not found")
    exit(1)

print(f"Memory store found: {store_id}")

# Creating memories from files in memory_seeding directory
for file in os.listdir("memory_seeding"):
    if file.endswith(".md"):
        with open(f"memory_seeding/{file}", "r") as f:
            memory = f.read()
        client.beta.memory_stores.memories.create(
            store_id,
            path=f'/{file.lower()}',
            content=memory
        )
        print(f"Memory {file} stored")