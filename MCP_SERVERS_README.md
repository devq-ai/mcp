# 🚀 Available MCP Servers

This document provides comprehensive information about the currently running MCP (Model Context Protocol) servers available for AI agents and applications.

## 📊 Server Status Overview

**Currently Running**: 8 MCP Servers  
**Status**: ✅ Operational and Ready  
**Last Updated**: 2025-01-14

---

## 🟢 Active MCP Servers

### 1. **Filesystem Server** 
- **Status**: ✅ Running (Terminal ID: 6)
- **Transport**: stdio
- **Capabilities**: 
  - Secure file read/write operations
  - Directory listing and navigation
  - File content manipulation
- **Security**: Restricted to `/Users/dionedge/devqai/mcp` directory
- **Usage**: Standard MCP filesystem operations
- **Package**: `@modelcontextprotocol/server-filesystem`

### 2. **Git Server**
- **Status**: ✅ Running (Terminal ID: 37) 
- **Transport**: stdio
- **Capabilities**:
  - Git repository operations
  - Commit history access
  - Branch management
  - Repository status and diff operations
- **Package**: `mcp-server-git` (uvx)
- **Usage**: Version control operations within the repository

### 3. **GitHub Server**
- **Status**: ✅ Running (Terminal ID: 24)
- **Transport**: stdio  
- **Capabilities**:
  - GitHub API integration
  - Repository management
  - Issue and PR operations
  - GitHub-specific workflows
- **Package**: `@modelcontextprotocol/server-github`
- **Requirements**: GitHub Personal Access Token (if needed)

### 4. **Fetch Server**
- **Status**: ✅ Running (Terminal ID: 26)
- **Transport**: stdio
- **Capabilities**:
  - Web content fetching
  - HTTP/HTTPS requests
  - Content conversion for LLM usage
  - API endpoint access
- **Package**: `mcp-server-fetch` (uvx)
- **Usage**: External web resource access

### 5. **Memory Server**
- **Status**: ✅ Running (Terminal ID: 20)
- **Transport**: stdio
- **Capabilities**:
  - Knowledge graph-based persistent memory
  - Cross-session data persistence
  - Structured information storage
  - Memory retrieval and search
- **Package**: `@modelcontextprotocol/server-memory`
- **Usage**: Long-term memory and knowledge management

### 6. **Inspector (Debug Tool)**
- **Status**: ✅ Running (Terminal ID: 23)
- **Transport**: Web Interface + stdio
- **Web Interface**: http://127.0.0.1:6274
- **Auth URL**: http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=03b0d84b938387ef51406f53e212816bb1940f4e9af62edea6260cd1a3d2ca32
- **Proxy**: 127.0.0.1:6277
- **Capabilities**:
  - MCP server connection debugging
  - Protocol testing and validation
  - Server interaction monitoring
  - Development and troubleshooting
- **Package**: `@modelcontextprotocol/inspector`

### 7. **Bayes MCP Server** (Custom)
- **Status**: ✅ Running (Terminal ID: 15)
- **Transport**: HTTP API + MCP
- **Web Interface**: http://127.0.0.1:8000
- **Capabilities**:
  - Bayesian inference and analysis
  - Statistical modeling
  - A/B testing frameworks
  - Probability calculations
  - Data analysis workflows
- **Features**:
  - Logfire integration for observability
  - FastAPI web interface
  - Auto-reload enabled
  - Comprehensive statistical tools
- **Location**: `/Users/dionedge/devqai/mcp/mcp-servers/bayes-mcp`

### 8. **Logfire MCP Server** (Custom)
- **Status**: ✅ Running (Terminal ID: 14)
- **Transport**: stdio
- **Capabilities**:
  - Observability and monitoring
  - Structured logging
  - Distributed tracing
  - Performance monitoring
  - Application insights
- **Integration**: Works with Bayes MCP for enhanced observability
- **Location**: `/Users/dionedge/devqai/mcp/mcp-servers/logfire-mcp`

---

## 🔧 How to Connect to MCP Servers

### For AI Agents/Applications:

#### **stdio Transport Servers**:
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
    "github": {
      "command": "npx", 
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"
      }
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

#### **HTTP/Web Interface Servers**:
- **Bayes MCP**: Connect to `http://127.0.0.1:8000`
- **Inspector**: Access via `http://127.0.0.1:6274`

---

## 🛠 Server Management

### Check Server Status:
```bash
# List all running processes
ps aux | grep -E "(mcp|npx|uvx)"

# Check specific terminals (if using the launch system)
# Terminal IDs: 6, 14, 15, 20, 23, 24, 26, 37
```

### Restart Servers:
```bash
# Filesystem
npx -y @modelcontextprotocol/server-filesystem .

# Git  
uvx mcp-server-git

# GitHub
npx -y @modelcontextprotocol/server-github

# Fetch
uvx mcp-server-fetch

# Memory
npx -y @modelcontextprotocol/server-memory

# Inspector
npx -y @modelcontextprotocol/inspector

# Bayes MCP (Custom)
cd /Users/dionedge/devqai/mcp/mcp-servers/bayes-mcp
source venv/bin/activate
python bayes_mcp.py

# Logfire MCP (Custom)  
cd /Users/dionedge/devqai/mcp/mcp-servers/logfire-mcp
source venv/bin/activate
python src/logfire-fastapi.py
```

---

## 📋 Capabilities Summary

| Server | File Ops | Web Access | Git Ops | Memory | Stats | Debug | Monitoring |
|--------|----------|------------|---------|--------|-------|-------|------------|
| Filesystem | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Git | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| GitHub | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Fetch | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Memory | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Inspector | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Bayes MCP | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Logfire MCP | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 🔐 Security & Access

- **Filesystem Server**: Restricted to project directory only
- **GitHub Server**: Requires valid GitHub token for full functionality  
- **Web Interfaces**: Running on localhost (127.0.0.1)
- **Inspector**: Has authentication token for secure access

---

## 📞 Support & Troubleshooting

### Common Issues:
1. **Server Not Responding**: Check if process is still running
2. **Permission Errors**: Verify file/directory permissions
3. **Network Issues**: Ensure localhost ports are available
4. **Token Issues**: Verify GitHub token is valid and has required permissions

### Logs Location:
- **Bayes MCP**: Check FastAPI logs at http://127.0.0.1:8000
- **Logfire MCP**: Integrated observability dashboard
- **Inspector**: Web interface provides detailed connection logs

---

*Last Updated: 2025-01-14*  
*Total Active Servers: 8*  
*Repository: /Users/dionedge/devqai/mcp*
