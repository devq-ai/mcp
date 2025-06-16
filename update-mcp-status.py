#!/usr/bin/env python3
"""
MCP Status Updater
Updates the MCP server status JSON file for the status page
"""

import os
import json
import subprocess
import time
from datetime import datetime
import sys

# Configuration
MCP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVERS_DIR = os.path.join(MCP_BASE_DIR, "mcp-servers")
STATUS_FILE = os.path.join(MCP_BASE_DIR, "mcp-status-site", "mcp-status.json")
CONFIG_FILE = os.path.join(MCP_BASE_DIR, "mcp-servers.json")

def check_process_running(process_name):
    """Check if a process is running by name"""
    try:
        output = subprocess.check_output(
            f"ps aux | grep -E '{process_name}' | grep -v grep", 
            shell=True, 
            text=True
        )
        return len(output.strip()) > 0
    except subprocess.CalledProcessError:
        return False

def get_server_configs():
    """Get server configurations from mcp-servers.json"""
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        
        return data.get("mcp_servers", {})
    except Exception as e:
        print(f"Error loading server configurations: {str(e)}")
        return {}

def update_status():
    """Update the status JSON file"""
    # Get server configurations
    server_configs = get_server_configs()
    
    # Prepare status data
    servers = []
    for server_name, config in server_configs.items():
        # Define process patterns to check
        patterns = [
            f"mcp-server-{server_name}",
            f"{server_name}-mcp",
            f"{server_name}_mcp",
            f"{server_name}.py",
            # Add the command and first arg as a pattern
            f"{config.get('command')} {' '.join(config.get('args', [])[:1])}"
        ]
        
        # Check if server is running
        status = "offline"
        for pattern in patterns:
            if check_process_running(pattern):
                status = "online"
                break
        
        # Check if server is installed locally
        is_local = os.path.exists(os.path.join(MCP_SERVERS_DIR, server_name))
        
        # Add server to list
        servers.append({
            "name": server_name,
            "status": status,
            "description": config.get("description", ""),
            "repository": config.get("repository", ""),
            "is_local": is_local,
            "last_checked": datetime.now().isoformat()
        })
    
    # Sort servers by name
    servers.sort(key=lambda s: s["name"])
    
    # Count online servers
    online_count = sum(1 for server in servers if server["status"] == "online")
    
    # Create status data
    status_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_servers": len(servers),
            "online_servers": online_count
        },
        "servers": servers
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    
    # Write status file
    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Updated status file with {len(servers)} servers ({online_count} online)")

if __name__ == "__main__":
    update_status()