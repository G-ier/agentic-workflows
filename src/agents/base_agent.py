import os
from typing import Any, Dict, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

class BaseAgent(BaseModel):
    """Base class for all trainable LLM agents."""
    name: str
    model_name: str = "gpt-4.1"
    temperature: float = 0.7
    memory: ConversationBufferMemory = None
    llm: ChatOpenAI = None
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([
                ("system", "You are a specialized digital marketing agent that can accurately provide good online marketing guidelines and judge the compliance of an ad."),
                ("human", "{input}")
            ])
    training_data: List[Dict[str, str]] = []
    
    def __init__(self, **data):
        # Initialize the base model with all data first
        super().__init__(**data)
        
        # Extract only the parameters we need for ChatOpenAI
        llm_params = {
            "model": self.model_name,
            "api_key": os.getenv("OPENAI_API_KEY") # Use api_key instead of openai_api_key
        }
        
        # Initialize ChatOpenAI with only the required parameters
        self.llm = ChatOpenAI(**llm_params)
        self.memory = ConversationBufferMemory()
        
        # Initialize an empty training data list if not provided
        if self.training_data is None:
            self.training_data = []
        
        # Initialize a basic prompt template if not provided
        if self.prompt_template is None:
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant specialized in providing information."),
                ("human", "{input}")
            ])
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the agent."""
        # Save LLM state to a text file
        llm_state = {
            "name": self.name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "memory": self.memory.dict()
        }
        
        # Create a filename based on the agent's name
        filename = f"{self.name}_llm_state.txt"
        
        # Write the LLM state to the file
        with open(filename, 'w') as f:
            for key, value in llm_state.items():
                f.write(f"{key}: {value}\n")

        print(f"LLM state saved to {filename}")
    
    def load_state(self, filename: str) -> None:
        """Load a saved state into the agent."""
        state = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    key, value = line.split(':', 1)  # Split on first colon only
                    key = key.strip()
                    value = value.strip()
                    state[key] = value

        # Convert string values to appropriate types
        self.name = state["name"]
        self.model_name = state["model_name"]
        self.temperature = float(state["temperature"])
        
        # Initialize new memory
        self.memory = ConversationBufferMemory()
        
        # Parse and load memory messages if they exist
        if "memory" in state:
            try:
                # Convert the string representation of memory back to a dictionary
                import ast
                memory_dict = ast.literal_eval(state["memory"])
                
                # Add messages back to memory
                if "chat_memory" in memory_dict and "messages" in memory_dict["chat_memory"]:
                    for msg in memory_dict["chat_memory"]["messages"]:
                        if msg["type"] == "human":
                            self.memory.chat_memory.add_user_message(msg["content"])
                        elif msg["type"] == "ai":
                            self.memory.chat_memory.add_ai_message(msg["content"])
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not load memory state: {e}")

        print(f"LLM state loaded from {filename}")

    # trains by giving prompts and answers
    def train(self, training_data: List[Dict[str, str]]) -> None:
        """Train the agent with specific examples."""
        self.training_data.extend(training_data)
        
        # Create a training prompt with examples
        training_prompt = self.prompt_template.format_messages(
            input="Here are some training examples:\n" + 
                  "\n".join([f"Input: {d['input']}\nOutput: {d['output']}" 
                            for d in training_data])
        )
        
        # Fine-tune the model with the training data
        # Note: In a real implementation, you would use OpenAI's fine-tuning API
        # or implement a more sophisticated training mechanism
        self.llm(training_prompt)

    # Inject additional or initial knowledge into the agent.
    # The init_prompt: What are guidelines for the online marketing ads? Draw inspiration from the following list of few guidelines:
    # The complete_prompt: Here is a complete guideline for the online marketing ads:
    def inject_knowledge(self, init_prompt: str, complete_prompt: str, knowledge_init: str, knowledge_complete: str) -> None:
        """Inject additional or initial knowledge into the agent."""

        self.training_data.append({
            "input": init_prompt + "\n" + complete_prompt,
            "output": knowledge_init + "\n" + knowledge_complete
        })

        # Create the knowledge prompt
        knowledge_prompt = self.prompt_template.format_messages(
            input="These are the most important rules and examples of online marketing guidelines:\n" + 
                  "\n".join([f"Input: {init_prompt} {complete_prompt} \nOutput: {knowledge_init} {knowledge_complete}"])
        )

        # Inject the knowledge into the agent
        self.llm(knowledge_prompt)
    
    def process_input(self, input_text: str) -> str:
        """Process input with specialized handling."""
        # Add context from training data if relevant
        context = self._get_relevant_context(input_text)
        
        # Format the prompt with context
        messages = self.prompt_template.format_messages(
            input=f"Context: {context}\nInput: {input_text}"
        )
        
        # Generate response
        response = self.llm(messages)

        return response.content
    
    def _get_relevant_context(self, input_text: str) -> str:
        """Get relevant context from training data."""
        # Simple implementation - in a real scenario, you would use
        # more sophisticated similarity matching
        relevant_examples = [
            d for d in self.training_data 
            if any(word in input_text.lower() for word in d['input'].lower().split())
        ]
        return "\n".join([f"Example: {d['input']} -> {d['output']}" 
                         for d in relevant_examples[:3]]) 
    
    # Possibly separate process - or separate agent
    def extract_text_from_images(image_paths, prompt=None):
        """
        Extracts text from images using a vision model.
        
        Args:
            image_paths (list): List of paths to image files
            prompt (str, optional): Specific instructions for the model
                                   about what text to extract
        
        Returns:
            str: Extracted text from the images
        """
        try:

            
            # Default prompt if none provided
            if prompt is None:
                prompt = "Extract all text visible in these images."
            
            # Get API key from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "Error: No API key found. Set the OPENAI_API_KEY environment variable."
            
            # Prepare images for the API request
            image_contents = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    })
            
            # Prepare the payload for the API request
            payload = {
                "model": "gpt-4-vision-preview",  # or another vision-capable model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *image_contents
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            # Send the request to the API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json=payload
            )
            
            # Parse the response
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {result.get('error', {}).get('message', 'Unknown error')}"
        
        except Exception as e:
            return f"Error processing images: {str(e)}"
    