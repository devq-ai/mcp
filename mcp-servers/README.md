# MCP Servers Directory

This directory contains local installations of Model Context Protocol (MCP) servers that are registered in the main tools registry.

## Directory Structure

Each MCP server should be installed in its own subdirectory:

```
mcp-servers/
├── bayes-mcp/          # Bayesian inference server
├── browser-tools-mcp/  # Browser automation toolkit
├── github-mcp/         # GitHub API integration
└── ...
```

## Installation

To install a new MCP server:

1. Create a new subdirectory for the server
2. Clone the repository or copy the implementation files
3. Follow the server-specific installation instructions
4. Update the main tools registry with the local path

## Active Servers

The following MCP servers are currently installed and configured:

| Server | Status | Local Path | Documentation |
|--------|--------|------------|---------------|
| bayes-mcp | ✅ Active | ../bayes/ | [Bayes README](../bayes/README.md) |

## Adding New Servers

When adding a new server:

1. Create an installation script in this directory
2. Document the server configuration in this README
3. Update the enabled status in the tools registry
4. Test the server connection with the TestModel

## Server Configuration

Each server should include:

- A configuration file (if applicable)
- Documentation for API endpoints
- Authentication setup instructions
- Testing scripts or examples

## Development Guidelines

When developing or modifying MCP servers:

1. Follow the [common development rules](/rules/common_rules.md)
2. Adhere to the [MCP specification](../spec.md)
3. Include proper error handling and logging
4. Document all API endpoints and parameters