## Alfred Agent 

This Agent will need to handle alot of tasks for throwing the Gotham Party of the century. 🦇

Alfred will need to have various tools:

1. RAG (Retrieval Augmented Generation)
2. External API Access For Data
3. Web Search

All of these will be required to keep Alfred on top of his game to ensure accurate weather forecasting for the right time to start the fireworks show, and also be up to date with the latest in cultural trends and science. but Alfred needs to also know all of the guests so keeping their details within a database is easier for quick recall.

### Project Structure

`tools.py` holds all of the developed tools for Alfred
`retriever.py` holds the retrieval logic for the RAG system's data parsing requirements
`app.py` integrates all components together to build the fully functional tools for Alfred.

The Guestbook and dataset is located on HuggingFace here: [Guestbook Dataset](https://huggingface.co/datasets/agents-course/unit3-invitees)

[HuggingFace DataSets Package](https://huggingface.co/docs/datasets/index)

**Install Datasets**

`uv add datasets`

looking over the dataset there are 4 columns for RAG:

| Column | Details |
| -------|--------|
| Name | Name of the guest for quick recall |
| Relation | relationship to the host of the part Mr. Wayne |
| Description | a brief description of the guest for small reference and conversation | 
| email | for sharing any interesting articles or links you fancy or keep in contact  |


