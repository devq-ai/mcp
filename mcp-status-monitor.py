#!/usr/bin/env python3
"""
MCP Server Status Monitor
Generates a JSON status file for all MCP servers and uploads it to a static website.
"""

import json
import os
import subprocess
import datetime
import requests
from pathlib import Path

# Configuration
MCP_SERVERS_DIR = Path("./mcp-servers")
OUTPUT_JSON = Path("./mcp-status.json")
WEBHOOK_URL = os.environ.get("STATUS_WEBHOOK_URL", "")  # Set this in your environment

def check_process_running(process_name):
    """Check if a process is running by name."""
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
    """Get all server configurations from the mcp-servers directory."""
    servers = []
    
    # Check main mcp-servers.json file
    try:
        with open("mcp-servers.json", "r") as f:
            main_config = json.load(f)
            for name, config in main_config.get("mcp_servers", {}).items():
                servers.append({
                    "name": name,
                    "description": config.get("description", ""),
                    "repository": config.get("repository", ""),
                    "status": "unknown"
                })
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading mcp-servers.json: {e}")
    
    # Check individual server directories
    for server_dir in MCP_SERVERS_DIR.glob("*"):
        if not server_dir.is_dir():
            continue
            
        config_file = server_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    
                # Find if this server is already in our list
                server_name = config.get("name", server_dir.name)
                existing = next((s for s in servers if s["name"] == server_name), None)
                
                if existing:
                    # Update existing entry
                    existing["repository"] = config.get("repository", existing["repository"])
                    existing["description"] = config.get("description", existing["description"])
                else:
                    # Add new entry
                    servers.append({
                        "name": server_name,
                        "description": config.get("description", ""),
                        "repository": config.get("repository", ""),
                        "status": "unknown"
                    })
            except json.JSONDecodeError:
                pass
    
    return servers

def check_server_status(servers):
    """Check the status of each server."""
    for server in servers:
        # Define process patterns to check for each server
        name = server["name"]
        patterns = [
            f"mcp-server-{name}",
            f"{name}-mcp",
            f"{name}_mcp",
            f"{name}.py"
        ]
        
        # Check if any pattern matches a running process
        server["status"] = "offline"
        for pattern in patterns:
            if check_process_running(pattern):
                server["status"] = "online"
                break
                
        # Add timestamp
        server["last_checked"] = datetime.datetime.now().isoformat()
    
    return servers

def main():
    """Main function to check status and generate JSON."""
    servers = get_server_configs()
    servers = check_server_status(servers)
    
    # Add metadata
    status_data = {
        "servers": servers,
        "metadata": {
            "total_servers": len(servers),
            "online_servers": sum(1 for s in servers if s["status"] == "online"),
            "generated_at": datetime.datetime.now().isoformat(),
            "hostname": os.uname().nodename
        }
    }
    
    # Write to file
    with open(OUTPUT_JSON, "w") as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Status file generated at {OUTPUT_JSON}")
    
    # Send to webhook if configured
    if WEBHOOK_URL:
        try:
            response = requests.post(
                WEBHOOK_URL,
                json=status_data,
                headers={"Content-Type": "application/json"}
            )
            print(f"Status sent to webhook: {response.status_code}")
        except Exception as e:
            print(f"Error sending to webhook: {e}")

if __name__ == "__main__":
    main()
