#!/usr/bin/env python3
"""
MCP Registry Update Tool

This script verifies and updates the MCP tools registry,
checking for local availability of servers and adding metadata.
"""

import os
import re
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Configuration
REGISTRY_FILE = "../tools.md"
SERVERS_DIR = "."
ROOT_DIR = "/Users/dionedge/devqai/mcp"

def read_registry() -> str:
    """Read the current registry file."""
    registry_path = os.path.join(os.path.dirname(__file__), REGISTRY_FILE)
    with open(registry_path, 'r') as f:
        return f.read()

def parse_registry(content: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse the registry markdown into a structured format."""
    categories = {}
    current_category = None
    current_tools = []
    
    lines = content.split('\n')
    for line in lines:
        # Detect category headers
        if line.startswith('### ') and 'count=' in line:
            if current_category:
                categories[current_category] = current_tools
            
            # Extract category name
            category_match = re.match(r'### (.*?) \(count=(\d+)\)', line)
            if category_match:
                current_category = category_match.group(1)
                current_tools = []
        
        # Parse tool table rows
        elif line.startswith('| **') and ' | ' in line:
            parts = line.split('|')
            if len(parts) >= 5:
                tool_name = parts[1].strip().replace('**', '')
                tool_data = {
                    'name': tool_name,
                    'description': parts[2].strip() if len(parts) > 2 else '',
                    'enabled': 'true' in parts[3].lower() if len(parts) > 3 else False,
                    'reference': parts[4].strip() if len(parts) > 4 else '',
                }
                
                # Add repository if available (for MCP Servers)
                if len(parts) > 5:
                    tool_data['repository'] = parts[2].strip()
                    tool_data['description'] = parts[3].strip()
                
                current_tools.append(tool_data)
    
    # Add the last category
    if current_category:
        categories[current_category] = current_tools
    
    return categories

def check_server_availability() -> Dict[str, bool]:
    """Check which MCP servers are available locally."""
    servers_dir = os.path.join(os.path.dirname(__file__), SERVERS_DIR)
    server_status = {}
    
    # Check MCP servers directory
    for item in os.listdir(servers_dir):
        if os.path.isdir(os.path.join(servers_dir, item)) and 'mcp' in item.lower():
            server_name = item.replace('-', '_').lower()
            server_status[server_name] = True
    
    # Check bayes-mcp link
    bayes_mcp_path = os.path.join(servers_dir, 'bayes-mcp')
    if os.path.exists(bayes_mcp_path) and (os.path.isdir(bayes_mcp_path) or os.path.islink(bayes_mcp_path)):
        server_status['bayes_mcp'] = True
    
    return server_status

def update_tool_counts(content: str, categories: Dict[str, List[Dict[str, str]]]) -> str:
    """Update tool counts in TOC and category headers."""
    for category, tools in categories.items():
        # Update TOC count
        toc_pattern = rf'- \[{re.escape(category)}\]\(#.*?\)( \d+)?'
        content = re.sub(toc_pattern, f'- [{category}](#{"".join(category.lower().split())}) {len(tools)}', content)
        
        # Update category header count
        header_pattern = rf'### {re.escape(category)} \(count=\d+\)'
        content = re.sub(header_pattern, f'### {category} (count={len(tools)})', content)
    
    return content

def mark_available_servers(categories: Dict[str, List[Dict[str, str]]], server_status: Dict[str, bool]) -> Dict[str, List[Dict[str, str]]]:
    """Mark servers as available based on local checks."""
    # Only modify MCP Servers category
    if "MCP Servers" in categories:
        for tool in categories["MCP Servers"]:
            tool_name = tool['name'].replace('-', '_').lower()
            if tool_name in server_status and server_status[tool_name]:
                tool['local_available'] = True
                tool['local_path'] = f"mcp-servers/{tool['name']}"
            else:
                tool['local_available'] = False
    
    return categories

def generate_updated_registry(categories: Dict[str, List[Dict[str, str]]]) -> str:
    """Generate updated registry markdown."""
    content = []
    
    # Add header
    content.append("# Agentical MCP Registry & Agent Framework\n")
    
    # Add TOC
    content.append("## 📋 Table of Contents\n")
    content.append("- [Tools Registry](#tools-registry)")
    for category in categories:
        content.append(f"  - [{category}](#{category.lower().replace(' ', '-')}) {len(categories[category])}")
    content.append("")
    
    # Add Tools Registry header
    content.append("## 🛠 Tools Registry\n")
    
    # Add each category
    for category, tools in categories.items():
        content.append(f"### {category} (count={len(tools)})\n")
        
        # Add category description if we have one
        if category == "Anthropic Core Tools":
            content.append("Built-in tools available in Claude 4 Sonnet for enhanced AI capabilities.\n")
        elif category == "Core Agent Tools":
            content.append("Essential tools for agent-based operations and enterprise workflows.\n")
        elif category == "MCP Servers":
            content.append("Production-ready Model Context Protocol servers for specialized capabilities.\n")
        elif category == "Pydantic AI Tools":
            content.append("Type-safe AI tools with Pydantic validation and structured output.\n")
        elif category == "Zed MCP Tools":
            content.append("Zed editor integration tools for seamless development workflows.\n")
        elif category == "Optional Tools":
            content.append("Additional tools that can be enabled as needed.\n")
        
        # Add table header based on category
        if category == "MCP Servers":
            content.append("| Tool | Repository | Description | Enabled | Reference |")
            content.append("|------|------------|-------------|---------|-----------|")
        elif category == "Optional Tools":
            content.append("| Tool | Repository | Description | Enabled |")
            content.append("|------|------------|-------------|---------|")
        else:
            content.append("| Tool | Description | Enabled | Reference |")
            content.append("|------|-------------|---------|-----------|")
        
        # Add tools
        for tool in tools:
            if category == "MCP Servers":
                # Add local availability indicator
                enabled_status = "✅ `true`" if tool.get('enabled', False) else "❌ `false`"
                if tool.get('local_available'):
                    enabled_status += " 🏠"  # Add house emoji for locally available
                
                content.append(f"| **{tool['name']}** | {tool.get('repository', '')} | {tool['description']} | {enabled_status} | {tool.get('reference', '')} |")
            elif category == "Optional Tools":
                enabled_status = "✅ `true`" if tool.get('enabled', False) else "❌ `false`"
                content.append(f"| **{tool['name']}** | {tool.get('repository', '')} | {tool['description']} | {enabled_status} |")
            else:
                enabled_status = "✅ `true`" if tool.get('enabled', False) else "❌ `false`"
                content.append(f"| **{tool['name']}** | {tool['description']} | {enabled_status} | {tool.get('reference', '')} |")
        
        content.append("")
    
    return "\n".join(content)

def main():
    try:
        # Read current registry
        registry_content = read_registry()
        
        # Parse the registry
        categories = parse_registry(registry_content)
        
        # Check server availability
        server_status = check_server_availability()
        print(f"Found {len(server_status)} locally available MCP servers:")
        for server, available in server_status.items():
            print(f"  - {server}: {'Available' if available else 'Not available'}")
        
        # Mark available servers
        categories = mark_available_servers(categories, server_status)
        
        # Update tool counts
        registry_content = update_tool_counts(registry_content, categories)
        
        # Generate updated registry (optional)
        # updated_content = generate_updated_registry(categories)
        
        # Save updated registry (optional)
        # registry_path = os.path.join(os.path.dirname(__file__), REGISTRY_FILE)
        # with open(registry_path, 'w') as f:
        #     f.write(updated_content)
        
        print("\nRegistry verification complete!")
        print(f"Total tools: {sum(len(tools) for tools in categories.values())}")
        for category, tools in categories.items():
            print(f"  - {category}: {len(tools)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())