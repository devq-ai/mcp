#!/usr/bin/env python3
"""
MCP Server API - Simple API server to manage MCP servers
"""

import os
import json
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import socket
import time

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
    if server_name in running_servers and running_servers[server