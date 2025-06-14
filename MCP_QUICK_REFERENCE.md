# 🚀 MCP Servers Quick Reference

## Currently Running (8 Servers)

### 📁 **Core Servers**
```bash
# Filesystem - File operations
npx -y @modelcontextprotocol/server-filesystem .

# Git - Version control  
uvx mcp-server-git

# GitHub - Repository management
npx -y @modelcontextprotocol/server-github

# Fetch - Web content access
uvx mcp-server-fetch

# Memory - Persistent storage
npx -y @modelcontextprotocol/server-memory
```

### 🌐 **Web Interfaces**
- **Inspector**: http://127.0.0.1:6274 (Debug tool)
- **Bayes MCP**: http://127.0.0.1:8000 (Statistical analysis)

### 🔧 **Custom Python Servers**
```bash
# Bayes MCP - Statistical analysis
cd /Users/dionedge/devqai/mcp/mcp-servers/bayes-mcp
source venv/bin/activate && python bayes_mcp.py

# Logfire MCP - Observability  
cd /Users/dionedge/devqai/mcp/mcp-servers/logfire-mcp
source venv/bin/activate && python src/logfire-fastapi.py
```

## 📋 MCP Configuration for Claude Desktop

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "cwd": "/Users/dionedge/devqai/mcp"
    },
    "git": {
      "command": "uvx", 
      "args": ["mcp-server-git"],
      "cwd": "/Users/dionedge/devqai/mcp"
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

## 🎯 Quick Capabilities

| What You Need | Use This Server |
|---------------|-----------------|
| Read/write files | **filesystem** |
| Git operations | **git** |
| GitHub API calls | **github** |  
| Web scraping/APIs | **fetch** |
| Remember things | **memory** |
| Debug MCP issues | **inspector** |
| Statistical analysis | **bayes-mcp** |
| Monitor performance | **logfire-mcp** |

## 🔍 Status Check
```bash
# Quick status check
ps aux | grep -E "(mcp|npx|uvx)" | grep -v grep
```

---
*8 servers ready • Updated 2025-01-14*
