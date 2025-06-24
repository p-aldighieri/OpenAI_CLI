import requests
import json

#!/usr/bin/env python3
"""
OpenAI Pro CLI Tool

A command-line interface for interacting with OpenAI's advanced models including O1 and O3 Pro.
This script provides easy access to OpenAI's API with support for context passing and reasoning effort control.

Last Updated: 2025-06-24
Processes: Text queries, optional context files or text
Columns/Types: N/A (text processing tool)

Usage:
    python openaipro.py "Your query here" --context file.txt --model o1-preview --reasoning-effort high
"""

import os
import argparse
import logging
from openai import OpenAI
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_arguments():
    # Default model set to O3 Pro
    default_model = 'o3-pro'
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Interact with OpenAI's advanced models via command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s "What is machine learning?"
  %(prog)s "Analyze this code" --context myfile.py
  %(prog)s "Complex reasoning task" --model o3-pro --reasoning-effort high"""
    )
    
    parser.add_argument('query', type=str, help='The query text to send to OpenAI')
    parser.add_argument('--context', type=str, help='Optional context text or path to context file')
    parser.add_argument('--model', type=str, default=default_model, 
                       help='Model to use (default: o3-pro, options: o3-pro, o1-preview, o1-mini)')
    parser.add_argument('--reasoning-effort', type=str, choices=['low', 'medium', 'high'], 
                       default='medium', help='Reasoning effort level for O1 models')
    parser.add_argument('--max-tokens', type=int, default=2000, 
                       help='Maximum tokens in response (default: 2000)')
    parser.add_argument('--temperature', type=float, default=0.7, 
                       help='Temperature for response randomness (0.0-2.0, default: 0.7)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def get_context(context_input):
    """Get context from file or direct text input."""
    if not context_input:
        return ""
    
    # Check if it's a file path
    if os.path.isfile(context_input):
        logger.info(f"Reading context from file: {context_input}")
        try:
            with open(context_input, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading context file: {e}")
            return ""
    else:
        logger.info("Using direct context text")
        return context_input

def call_openai_api(client, query, context, model, reasoning_effort, max_tokens, temperature, verbose=False):
    """Call OpenAI API with the specified parameters."""
    
    # Combine query and context
    full_prompt = query
    if context:
        full_prompt = f"Context:\n{context}\n\nQuery: {query}"
    
    logger.info(f"Using model: {model}")
    logger.debug(f"Full prompt length: {len(full_prompt)} characters")
    
    try:
        # Handle O3 models
        if model.startswith('o3'):
            if model == 'o3-pro':
                # O3 Pro uses direct API call to responses endpoint
                logger.info("Using O3 Pro with direct API call and streaming")
                
                headers = {
                    'Authorization': f'Bearer {client.api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'model': model,
                    'instructions': 'You are a helpful assistant.',
                    'input': full_prompt,
                    'max_output_tokens': max_tokens,
                    'stream': False
                }
                
                try:
                    response = requests.post('https://api.openai.com/v1/responses', headers=headers, json=data)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    
                    result = response.json()
                    logger.debug(f"Full response: {result}")
                    
                    # Extract text from the response
                    if 'output' in result and len(result['output']) > 0:
                        for output_item in result['output']:
                            if output_item.get('type') == 'message' and 'content' in output_item:
                                for content_item in output_item['content']:
                                    if content_item.get('type') == 'output_text':
                                        return content_item.get('text', '')
                    
                    return "No response content found"
                
                except requests.exceptions.RequestException as e:
                    logger.error(f"Direct API call failed: {e}")
                    raise
            else:
                # O3 mini uses chat completions endpoint
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    max_completion_tokens=max_tokens
                )
                return response.choices[0].message.content
        elif model.startswith('o1'):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise

def main():
    """Main function to execute the CLI tool."""
    args = setup_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error('OPENAI_API_KEY not found in environment variables')
        logger.error('Please set your OpenAI API key: export OPENAI_API_KEY="your-key-here"')
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    logger.info('OpenAI client initialized')
    
    # Get context
    context_info = get_context(args.context)
    
    # Execute API call
    try:
        logger.info('Sending request to OpenAI...')
        result = call_openai_api(
            client, 
            args.query, 
            context_info, 
            args.model,
            args.reasoning_effort,
            args.max_tokens,
            args.temperature,
            args.verbose
        )
        
        logger.info('Response received successfully')
        print("\n" + "="*50)
        print("OpenAI Response:")
        print("="*50)
        print(result)
        print("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
