import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from agents.specialized_agent import SpecializedAgent
import json
from datetime import datetime

class AgentCron:
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
    
    def process_batch(self, input_texts: list) -> list:
        """Process a batch of requests and return the responses."""
        results = []
        for text in input_texts:
            try:
                response = self.agent.process_input(text)
                results.append({
                    "status": "success",
                    "input": text,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "input": text,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        return results
    
    def save_results(self, results: list, output_file: str):
        """Save the processing results to a file."""
        with open(output_file, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    
    def run_cron_job(self, input_file: str, output_file: str):
        """Run the agent as a cron job."""
        print(f"Starting cron job at {datetime.now().isoformat()}")
        
        try:
            # Read input file
            with open(input_file, 'r') as f:
                input_texts = [line.strip() for line in f if line.strip()]
            
            # Process inputs
            results = self.process_batch(input_texts)
            
            # Save results
            self.save_results(results, output_file)
            
            print(f"Cron job completed successfully at {datetime.now().isoformat()}")
            
        except Exception as e:
            print(f"Error in cron job: {str(e)}")

def main():
    # Create the agent cron job
    agent_cron = AgentCron()
    
    # Example usage
    input_file = "input_questions.txt"
    output_file = "output_responses.txt"
    
    # Run the cron job
    agent_cron.run_cron_job(input_file, output_file)

if __name__ == "__main__":
    main() 