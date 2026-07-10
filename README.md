# AgentOS

A collection of hands-on exercises and small projects for learning how to build AI agents with **LangChain** and **LangGraph**. This repo tracks the progression from a basic chatbot to more advanced agent patterns: tool use, ReAct-style reasoning, retrieval-augmented generation (RAG), and human-in-the-loop workflows.

## What's in here

| File | Description |
|---|---|
| `bot.py` | A minimal LangGraph chatbot: a single-node graph that sends user input to `gpt-4o` and prints the response. Good starting point for understanding `StateGraph`, nodes, and edges. |
| `drafter.py` | A "human-in-the-loop" writing assistant agent. Exposes `update` and `save` tools so the agent can revise a document based on user feedback and save it to a `.txt` file, looping until the user is satisfied. |
| `reActAgent.py` | An implementation of the ReAct (Reason + Act) agent pattern, where the model alternates between reasoning and calling tools. |
| `RAG_Agent.py` | A retrieval-augmented generation agent. Loads a PDF, chunks it, embeds it with OpenAI embeddings, stores it in a Chroma vector store, and lets the agent query it via a retriever tool to answer questions with citations. |
| `conditiona_agent.ipynb` | Notebook exercise on conditional edges / branching logic in a LangGraph graph. |
| `looping.ipynb` | Notebook exercise on building loops (repeated node execution) into a graph. |
| `multiple_inputs.ipynb` | Notebook exercise on handling agent state with multiple input fields. |
| `sequenntial_graph.ipynb` | Notebook exercise on chaining multiple nodes together in sequence. |
| `langraph.ipynb` | General LangGraph exploration/notes notebook. |
| `execersie_v.ipynb` | Additional practice exercise notebook. |
| `graph.png` | Exported visualization of one of the compiled agent graphs. |
| `meta_welcome.txt` / `recruiter_cold_email.txt` | Sample text used as input/output for the drafter and RAG exercises. |

## Getting started

### Prerequisites
- Python 3.10+
- An OpenAI API key

### Installation

```bash
git clone https://github.com/InjongWon/AgentOS.git
cd AgentOS
pip install langchain langchain-openai langchain-community langchain-chroma langgraph python-dotenv pypdf
```

Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=your-key-here
```

### Running an example

```bash
python bot.py
```

This starts a simple REPL-style chat loop in the terminal — type a message and press enter, or type `exit` to quit.

Other scripts (`drafter.py`, `reActAgent.py`, `RAG_Agent.py`) can be run the same way. Note that `RAG_Agent.py` expects a PDF path and a Chroma persistence directory to be set before it will run — update the relevant variables in the script first.

## Notes

This is just for langgraph/mcp orchestration repo so scripts are exploratory and may contain rough edges or incomplete pieces rather than production-ready code. Contributions, fixes, and suggestions are welcome via issues or pull requests.

## License

No license specified yet — all rights reserved by default until one is added.