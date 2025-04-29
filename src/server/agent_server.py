import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from agents.specialized_agent import SpecializedAgent
import json
from datetime import datetime

class AgentServer:
    def __init__(self, agent_config: Dict[str, Any] = None):
        load_dotenv()
        self.agent = SpecializedAgent(
            name="MarketingGuidelinesAgent",
            model_name="gpt-4-turbo-preview",
            temperature=0.7
        )
        self.load_initial_training()
        
    def load_initial_training(self):
        """Load initial training data for the agent."""
        training_data = [
            {
                "input": "What are the key requirements for Facebook ad content?",
                "output": "Facebook ad content must: 1) Be truthful and not misleading, 2) Avoid prohibited content (e.g., adult content, illegal products), 3) Follow community standards, 4) Include proper targeting, 5) Have clear call-to-actions, and 6) Comply with Facebook's advertising policies."
            },
            {
                "input": "How should I handle sensitive topics in ads?",
                "output": "When handling sensitive topics in ads: 1) Be respectful and considerate, 2) Avoid sensationalism, 3) Provide appropriate context, 4) Follow platform-specific guidelines, 5) Consider cultural sensitivities, and 6) Ensure compliance with relevant regulations."
            }
        ]
        self.agent.train(training_data)
    
    def process_request(self, input_text: str) -> Dict[str, Any]:
        """Process a single request and return the response."""
        try:
            response = self.agent.process_input(input_text)
            return {
                "status": "success",
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_as_service(self, interval: int = 60):
        """Run the agent as a continuous service."""
        print(f"Starting agent service. Checking for requests every {interval} seconds...")
        
        while True:
            try:
                # Here you would typically check for new requests
                # This could be from a queue, database, or API endpoint
                # For now, we'll just keep the service running
                time.sleep(interval)
            except KeyboardInterrupt:
                print("Shutting down agent service...")
                break
            except Exception as e:
                print(f"Error in agent service: {str(e)}")
                time.sleep(interval)

def main():
    # Create and run the agent server
    server = AgentServer()
    
    # Example: Run as a continuous service
    server.run_as_service(interval=60)

if __name__ == "__main__":
    main() 