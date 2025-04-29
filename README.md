# Trainable Agents with LangChain

This project provides a framework for creating trainable agents using LangChain and OpenAI's API. It includes a base agent class that can be extended to create specialized agents with custom training capabilities.

## Project Structure

```
.
├── src/
│   ├── agents/
│   │   ├── base_agent.py      # Base agent class
│   │   └── specialized_agent.py # Example specialized agent
│   ├── applications/
│   │   └── agent_example.py   # Example application
│   ├── server/
│   │   └── agent_server.py    # Server implementation
│   └── cron/
│       └── agent_cron.py      # Cron job implementation
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

The project provides several components:

1. `BaseAgent`: A foundation class for creating agents with basic conversation capabilities
2. `SpecializedAgent`: An example implementation showing how to extend the base agent
3. `AgentServer`: A server implementation for running agents as a service
4. `AgentCron`: A cron job implementation for scheduled agent tasks

Example applications can be found in `src/applications/`.

## Features

- Conversation memory
- Training capabilities
- State saving and loading
- Customizable model parameters
- Extensible architecture
- Server and cron job implementations

## Extending the Framework

To create your own specialized agent:

1. Create a new class that inherits from `BaseAgent`
2. Implement the `train` method with your specific training logic
3. Override `process_input` if you need custom input processing
4. Add any additional methods specific to your use case

## License

MIT 