import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import tiktoken
from pathlib import Path
from typing import Dict, List, Tuple


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def generate_text_file(target_tokens: int, output_path: str, encoding_name: str = "cl100k_base") -> None:
    """Generate a text file with approximately target_tokens tokens."""
    encoding = tiktoken.get_encoding(encoding_name)
    
    # Base text to repeat and pad
    base_text = "The quick brown fox jumps over the lazy dog. "
    base_text += "This is a sample paragraph for testing token counts. "
    base_text += "Additional content to help reach target token counts. "
    base_text += "More text here to ensure we can reach the desired token count. "
    base_text += "The file reading loop processes each line efficiently. "
    base_text += "Session resume scenarios require handling large inputs. "
    
    # Calculate how many repetitions needed
    test_text = ""
    while True:
        test_text += base_text
        tokens = count_tokens(test_text, encoding_name)
        if tokens >= target_tokens:
            break
    
    # Truncate to target (with some tolerance)
    while count_tokens(test_text, encoding_name) > target_tokens + 100:
        # Remove last sentence
        sentences = test_text.rsplit('. ', 1)
        if len(sentences) > 1:
            test_text = sentences[0] + '. '
        else:
            test_text = test_text[:int(len(test_text) * 0.9)]
            break
    
    # Ensure we're close to target
    final_tokens = count_tokens(test_text, encoding_name)
    if final_tokens > target_tokens + 50:
        # Trim more aggressively
        while count_tokens(test_text, encoding_name) > target_tokens:
            test_text = test_text[:int(len(test_text) * 0.95)]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(test_text)
    
    final_tokens = count_tokens(test_text, encoding_name)
    print(f"Generated: {output_path} ({final_tokens} tokens, target: {target_tokens})")


def generate_code_file(target_tokens: int, output_path: str, encoding_name: str = "cl100k_base") -> None:
    """Generate a Python code file with approximately target_tokens tokens."""
    encoding = tiktoken.get_encoding(encoding_name)
    
    # Generate code with comments to reach target tokens
    code_template = '''# Function to process a single item
def process_item(item):
    """Process a single item and return the result."""
    # Validate input
    if item is None:
        return None
    
    # Process the item
    result = item * 2
    return result


# Main processing function
def process_items(items):
    """Process a list of items."""
    results = []
    for item in items:
        # Process each item individually
        processed = process_item(item)
        if processed is not None:
            results.append(processed)
    return results


# Additional utility functions
def calculate_total(values):
    """Calculate the total of a list of values."""
    total = 0
    for value in values:
        total += value
    return total


def filter_positive(numbers):
    """Filter out negative numbers."""
    return [n for n in numbers if n >= 0]


# Class definition
class DataProcessor:
    """A class for processing data."""
    
    def __init__(self, data):
        self.data = data
        self.processed = []
    
    def process(self):
        """Process the stored data."""
        self.processed = process_items(self.data)
        return self.processed
    
    def get_results(self):
        """Return processed results."""
        return self.processed


# Main execution
if __name__ == "__main__":
    # Sample data
    sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Create processor and run
    processor = DataProcessor(sample_data)
    results = processor.process()
    
    # Output results
    print(f"Processed {len(results)} items")
    print(f"Total: {calculate_total(results)}")
    print(f"Positive filtered: {filter_positive(results)}")
'''
    
    # Repeat and pad to reach target
    test_code = ""
    while True:
        test_code += code_template + "\n"
        tokens = count_tokens(test_code, encoding_name)
        if tokens >= target_tokens:
            break
    
    # Trim to target
    while count_tokens(test_code, encoding_name) > target_tokens + 50:
        test_code = test_code[:int(len(test_code) * 0.95)]
    
    # Ensure we're close
    while count_tokens(test_code, encoding_name) > target_tokens:
        lines = test_code.split('\n')
        if len(lines) > 1:
            test_code = '\n'.join(lines[:-1])
        else:
            test_code = test_code[:int(len(test_code) * 0.9)]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(test_code)
    
    final_tokens = count_tokens(test_code, encoding_name)
    print(f"Generated: {output_path} ({final_tokens} tokens, target: {target_tokens})")


def generate_conversation_context(file_content: str, role: str = "user") -> str:
    """Generate a conversation context with system prompt, user message, and file content."""
    context = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. The user will provide you with file content and ask questions about it. Read the file carefully and provide accurate responses.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the content of a file I'm working with:

[FILE CONTENT START]
'''
    context += file_content
    context += '''[FILE CONTENT END]

How can I help you with this file? Please read and analyze the content above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I've read the file content you provided. The file contains:

'''
    return context


def main():
    """Generate all test data files."""
    config = load_config()
    
    text_dir = Path(config['test_data']['text_dir'])
    code_dir = Path(config['test_data']['code_dir'])
    
    # Create directories
    text_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate text files
    print("\n=== Generating Text Files ===")
    for target in config['test_params']['input_targets']:
        generate_text_file(
            target,
            str(text_dir / f"text_{target}.txt")
        )
    
    # Generate code files
    print("\n=== Generating Code Files ===")
    for target in config['test_params']['input_targets']:
        generate_code_file(
            target,
            str(code_dir / f"code_{target}.py")
        )
    
    # Generate conversation context files
    print("\n=== Generating Conversation Context Files ===")
    for target in config['test_params']['input_targets']:
        text_path = text_dir / f"text_{target}.txt"
        with open(text_path, 'r') as f:
            file_content = f.read()
        
        context = generate_conversation_context(file_content)
        context_path = text_dir / f"conversation_{target}.txt"
        with open(context_path, 'w') as f:
            f.write(context)
        
        tokens = count_tokens(context)
        print(f"Generated: {context_path} ({tokens} tokens with context)")
    
    print("\n=== Test Data Generation Complete ===")


if __name__ == "__main__":
    main()
