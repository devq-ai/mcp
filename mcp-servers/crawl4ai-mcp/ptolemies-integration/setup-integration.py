#!/usr/bin/env python3
"""
SurrealDB MCP + Ptolemies Integration Setup
"""

import os
import json
import argparse
from pathlib import Path

def setup_integration(config_path=None, ptolemies_path=None):
    """Set up the integration between SurrealDB MCP and Ptolemies."""
    print("Setting up SurrealDB MCP + Ptolemies integration...")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print("Config file not found!")
            return False
    
    # Determine Ptolemies path
    if not ptolemies_path:
        ptolemies_path = os.path.join(os.path.dirname(__file__), "ptolemies")
        if os.path.islink(ptolemies_path):
            ptolemies_path = os.readlink(ptolemies_path)
    
    if not os.path.exists(ptolemies_path):
        print(f"Ptolemies path not found: {ptolemies_path}")
        return False
    
    print(f"Using Ptolemies at: {ptolemies_path}")
    
    # Create example knowledge item
    knowledge_item = {
        "title": "SurrealDB MCP Integration",
        "content": "This is a test knowledge item for the SurrealDB MCP + Ptolemies integration.",
        "tags": ["mcp", "surrealdb", "integration", "test"],
        "metadata": {
            "source": "integration_setup",
            "timestamp": "2025-06-01T12:00:00Z"
        }
    }
    
    # Save example knowledge item
    example_path = os.path.join(os.path.dirname(__file__), "example_knowledge.json")
    with open(example_path, 'w') as f:
        json.dump(knowledge_item, f, indent=2)
    
    print(f"Created example knowledge item at: {example_path}")
    
    # Create symlink to integration in Ptolemies (if it doesn't exist)
    ptolemies_integrations_dir = os.path.join(ptolemies_path, "integrations")
    if not os.path.exists(ptolemies_integrations_dir):
        os.makedirs(ptolemies_integrations_dir, exist_ok=True)
    
    integration_symlink = os.path.join(ptolemies_integrations_dir, "surrealdb-mcp")
    if not os.path.exists(integration_symlink):
        os.symlink(os.path.dirname(__file__), integration_symlink)
        print(f"Created symlink in Ptolemies integrations directory")
    
    print("Integration setup complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up SurrealDB MCP + Ptolemies integration")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--ptolemies", help="Path to Ptolemies directory")
    args = parser.parse_args()
    
    setup_integration(args.config, args.ptolemies)
