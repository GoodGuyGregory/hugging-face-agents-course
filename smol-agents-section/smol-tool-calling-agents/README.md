## SmolAgents - Building Tool Calling Agents

#### Documentation

[Tool Calling](https://huggingface.co/learn/agents-course/unit2/smolagents/tools)

### Tool Building

`Tools` require the following to be relevant for the LLM to process and format the JSON responses to then be executed on your behalf.

* Name: What the tool is called.
* Tool Description: a defined scope and definition of the tool's functionality for the model, as provided context to allude to it's purpose for being utilized.

* Input Types: Tool arguments and their specific data types, `str, int, bool`, such as  `record: str`. a brief description is key too. 

`record (str): title of the record the user wants to fetch details about.`

* Output Type: what the tool will respond with, often a description is nice to have for clarity.

* `@tool` decorator for creating a subclass of `Tool` from the framework. This will indicate that the tool is a simple function the LLM can access. 

#### Tool Example







