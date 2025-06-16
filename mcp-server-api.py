#!/usr/bin/env python3
"""
MCP Server API - Simple API server to manage MCP servers
"""

import os
import json
import subprocess
import threading
import time
import signal
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import socket

# Configuration
PORT = 8080
MCP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVERS_DIR = os.path.join(MCP_BASE_DIR, "mcp-servers")
STATUS_FILE = os.path.join(MCP_BASE_DIR, "mcp-status-site", "mcp-status.json")

# Store running server processes
running_servers = {}

class MCPServerHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_OPTIONS(self):
        self._set_headers()
        
    def do_GET(self):
        if self.path == "/":
            # Serve the status page
            self._serve_file("mcp-status-site/index.html", "text/html")
        elif self.path.startswith("/mcp-status.json"):
            # Serve the status JSON file
            self._serve_file("mcp-status-site/mcp-status.json", "application/json")
        else:
            # Try to serve static files from mcp-status-site directory
            file_path = os.path.join("mcp-status-site", self.path.lstrip("/"))
            if os.path.exists(file_path) and os.path.isfile(file_path):
                content_type = self._get_content_type(file_path)
                self._serve_file(file_path, content_type)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")
    
    def _serve_file(self, file_path, content_type):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._set_headers(content_type)
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())
    
    def _get_content_type(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
        return content_types.get(ext, "application/octet-stream")
    
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        
        try:
            data = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            data = {}
        
        if self.path == "/api/start-server":
            self._handle_start_server(data)
        elif self.path == "/api/stop-server":
            self._handle_stop_server(data)
        elif self.path == "/api/install-server":
            self._handle_install_server(data)
        elif self.path == "/api/start-all-servers":
            self._handle_start_all_servers()
        else:
            self._set_headers()
            self.wfile.write(json.dumps({"error": "Unknown endpoint"}).encode())
    
    def _handle_start_server(self, data):
        server_name = data.get("server")
        if not server_name:
            self._set_headers()
            self.wfile.write(json.dumps({"error": "Server name is required"}).encode())
            return
        
        output = start_server(server_name)
        
        self._set_headers()
        self.wfile.write(json.dumps({"success": True, "output": output}).encode())
    
    def _handle_stop_server(self, data):
        server_name = data.get("server")
        if not server_name:
            self._set_headers()
            self.wfile.write(json.dumps({"error": "Server name is required"}).encode())
            return
        
        output = stop_server(server_name)
        
        self._set_headers()
        self.wfile.write(json.dumps({"success": True, "output": output}).encode())
    
    def _handle_install_server(self, data):
        server_name = data.get("server")
        if not server_name:
            self._set_headers()
            self.wfile.write(json.dumps({"error": "Server name is required"}).encode())
            return
        
        output = install_server(server_name)
        
        self._set_headers()
        self.wfile.write(json.dumps({"success": True, "output": output}).encode())
    
    def _handle_start_all_servers(self):
        output = start_all_servers()
        
        self._set_headers()
        self.wfile.write(json.dumps({"success": True, "output": output}).encode())

def start_server(server_name):
    """Start an MCP server"""
    # Check if server is already running
    if server_name in running_servers and running_servers[server_name].poll() is None:
        return f"Server '{server_name}' is already running"
    
    # Get server configuration from mcp-servers.json
    server_config = get_server_config(server_name)
    if not server_config:
        return f"Server '{server_name}' not found in configuration"
    
    # Prepare command
    command = server_config.get("command")
    args = server_config.get("args", [])
    cwd = server_config.get("cwd", MCP_SERVERS_DIR)
    env = os.environ.copy()
    
    # Add server-specific environment variables
    if "env" in server_config:
        for key, value in server_config["env"].items():
            # Handle environment variable substitution
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                # Check if there's a default value after a colon
                if ":-" in env_var:
                    env_var_name, default_value = env_var.split(":-", 1)
                    env[key] = os.environ.get(env_var_name, default_value)
                else:
                    env[key] = os.environ.get(env_var, "")
            else:
                env[key] = value
    
    # Start the server process
    try:
        full_command = [command] + args
        process = subprocess.Popen(
            full_command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Store the process
        running_servers[server_name] = process
        
        # Start a thread to read output
        output_thread = threading.Thread(
            target=read_process_output,
            args=(process, server_name),
            daemon=True
        )
        output_thread.start()
        
        # Update status file
        update_status_file()
        
        return f"Started server '{server_name}' with PID {process.pid}"
    except Exception as e:
        return f"Error starting server '{server_name}': {str(e)}"

def stop_server(server_name):
    """Stop an MCP server"""
    if server_name not in running_servers:
        return f"Server '{server_name}' is not running"
    
    process = running_servers[server_name]
    if process.poll() is not None:
        del running_servers[server_name]
        return f"Server '{server_name}' is not running"
    
    try:
        # Try to terminate gracefully first
        process.terminate()
        
        # Wait for a short time
        for _ in range(10):
            if process.poll() is not None:
                break
            time.sleep(0.1)
        
        # If still running, kill it
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        
        del running_servers[server_name]
        
        # Update status file
        update_status_file()
        
        return f"Stopped server '{server_name}'"
    except Exception as e:
        return f"Error stopping server '{server_name}': {str(e)}"

def install_server(server_name):
    """Install an MCP server"""
    server_dir = os.path.join(MCP_SERVERS_DIR, server_name)
    
    # Check if server directory exists
    if not os.path.exists(server_dir):
        try:
            os.makedirs(server_dir)
        except Exception as e:
            return f"Error creating directory for '{server_name}': {str(e)}"
    
    # Check for install script
    install_script = os.path.join(server_dir, "install.sh")
    if os.path.exists(install_script):
        try:
            # Make sure the script is executable
            os.chmod(install_script, 0o755)
            
            # Run the install script
            process = subprocess.Popen(
                ["bash", install_script],
                cwd=server_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            output, _ = process.communicate()
            
            if process.returncode == 0:
                # Update status file
                update_status_file()
                return f"Successfully installed '{server_name}':\n\n{output}"
            else:
                return f"Error installing '{server_name}' (exit code {process.returncode}):\n\n{output}"
        except Exception as e:
            return f"Error running install script for '{server_name}': {str(e)}"
    else:
        return f"No install script found for '{server_name}'"

def start_all_servers():
    """Start all available servers"""
    # Get all server configurations
    server_configs = get_all_server_configs()
    
    results = []
    for server_name in server_configs:
        # Skip servers that are already running
        if server_name in running_servers and running_servers[server_name].poll() is None:
            results.append(f"Server '{server_name}' is already running")
            continue
        
        # Start the server
        result = start_server(server_name)
        results.append(result)
    
    return "\n".join(results)

def read_process_output(process, server_name):
    """Read and log output from a server process"""
    log_dir = os.path.join(MCP_BASE_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{server_name}.log")
    
    with open(log_file, "a") as f:
        f.write(f"\n--- Server started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                f.write(line)
                f.flush()

def get_server_config(server_name):
    """Get configuration for a specific server"""
    all_configs = get_all_server_configs()
    return all_configs.get(server_name)

def get_all_server_configs():
    """Get configurations for all servers"""
    config_file = os.path.join(MCP_BASE_DIR, "mcp-servers.json")
    
    try:
        with open(config_file, "r") as f:
            data = json.load(f)
        
        return data.get("mcp_servers", {})
    except Exception as e:
        print(f"Error loading server configurations: {str(e)}")
        return {}

def update_status_file():
    """Update the status JSON file with current server status"""
    try:
        # Read current status file
        with open(STATUS_FILE, "r") as f:
            status_data = json.load(f)
        
        # Update server status
        for server in status_data.get("servers", []):
            server_name = server.get("name")
            
            if server_name in running_servers and running_servers[server_name].poll() is None:
                server["status"] = "online"
            else:
                server["status"] = "offline"
        
        # Update metadata
        online_count = sum(1 for server in status_data.get("servers", []) if server.get("status") == "online")
        status_data["metadata"]["online_servers"] = online_count
        status_data["metadata"]["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Write updated status file
        with open(STATUS_FILE, "w") as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        print(f"Error updating status file: {str(e)}")

def signal_handler(sig, frame):
    """Handle termination signals"""
    print("Shutting down...")
    
    # Stop all running servers
    for server_name in list(running_servers.keys()):
        stop_server(server_name)
    
    sys.exit(0)

def main():
    """Run the API server"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create server
    server = HTTPServer(("", PORT), MCPServerHandler)
    print(f"Starting MCP Server API on port {PORT}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped")

if __name__ == "__main__":
    main()
