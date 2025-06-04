#!/usr/bin/env python3
"""
Update the MCP tools registry to mark locally available servers.
"""

import os
import re
import json
from pathlib import Path

# Constants
TOOLS_MD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools.md")
SERVERS_DIR = os.path.abspath(os.path.dirname(__file__))

def get_local_servers():
    """Get list of locally available MCP servers."""
    local_servers = []
    for item in os.listdir(SERVERS_DIR):
        item_path = os.path.join(SERVERS_DIR, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if it's installed
            config_path = os.path.join(item_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        installed = config.get('local_installed', False)
                        if installed:
                            local_servers.append(item)
                        else:
                            # Check if it's a symlink (like bayes-mcp)
                            if os.path.islink(item_path):
                                local_servers.append(item)
                except:
                    pass
            elif os.path.islink(item_path):
                local_servers.append(item)
    return local_servers

def update_tools_registry():
    """Update the tools.md file to mark locally available servers."""
    if not os.path.exists(TOOLS_MD_PATH):
        print(f"Error: tools.md not found at {TOOLS_MD_PATH}")
        return
    
    # Get locally available servers
    local_servers = get_local_servers()
    print(f"Found {len(local_servers)} locally available servers: {', '.join(local_servers)}")
    
    # Read the current tools.md
    with open(TOOLS_MD_PATH, 'r') as f:
        content = f.read()
    
    # Find the MCP Servers section
    mcp_servers_section = re.search(r'### MCP Servers \(count=\d+\)(.*?)(?=### [A-Za-z]|\Z)', content, re.DOTALL)
    if not mcp_servers_section:
        print("Error: MCP Servers section not found in tools.md")
        return
    
    section = mcp_servers_section.group(1)
    
    # Process each server row
    for server in local_servers:
        # Normalize server name for matching
        server_match = server.replace('-', '[_\-]')
        # Look for the server row
        server_pattern = rf'\| \*\*{server_match}\*\* \|(.*?)\| (✅|❌) ["] \|'
        
        # Replace with local availability marker
        if re.search(server_pattern, section, re.IGNORECASE):
            modified_section = re.sub(
                server_pattern, 
                f'| **{server}** |\1| ✅  🏠 |', 
                section, 
                flags=re.IGNORECASE
            )
            # Update the content
            content = content.replace(section, modified_section)
            section = modified_section
            print(f"Updated {server} in registry with local availability marker")
    
    # Write the updated content
    with open(TOOLS_MD_PATH, 'w') as f:
        f.write(content)
    
    print(f"Updated tools.md with local server availability")

if __name__ == "__main__":
    update_tools_registry()
