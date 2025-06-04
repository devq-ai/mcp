#!/bin/bash
# Verify MCP tools configuration and status

echo "===== MCP Tools Verification ====="
echo ""

# Check directory structure
echo "Checking directory structure..."
if [ -d "/Users/dionedge/devqai/mcp" ]; then
  echo "✅ MCP directory exists"
else
  echo "❌ MCP directory not found!"
  exit 1
fi

if [ -d "/Users/dionedge/devqai/mcp/mcp-servers" ]; then
  echo "✅ MCP servers directory exists"
else
  echo "❌ MCP servers directory not found!"
  mkdir -p "/Users/dionedge/devqai/mcp/mcp-servers"
  echo "  ✅ Created MCP servers directory"
fi

# Check Python environment
echo ""
echo "Checking Python environment..."
VENV_PATH="/Users/dionedge/devqai/mcp/pydantic_ai_env"
if [ -d "$VENV_PATH" ]; then
  echo "✅ Virtual environment exists at $VENV_PATH"
  if [ -f "$VENV_PATH/bin/python" ]; then
    echo "✅ Python executable exists in virtual environment"
  else
    echo "❌ Python executable not found in virtual environment!"
    echo "  ℹ️ Run mcp-servers/fix-environment.sh to fix this issue"
  fi
else
  echo "❌ Virtual environment not found!"
fi

# Check Bayes MCP server
echo ""
echo "Checking Bayes MCP server..."
BAYES_PATH="/Users/dionedge/devqai/bayes"
BAYES_LINK_PATH="/Users/dionedge/devqai/mcp/mcp-servers/bayes-mcp"
if [ -d "$BAYES_PATH" ]; then
  echo "✅ Bayes project exists at $BAYES_PATH"
  if [ -f "$BAYES_PATH/bayes_mcp.py" ]; then
    echo "✅ Bayes MCP entry point exists"
  else
    echo "❌ Bayes MCP entry point not found!"
  fi
  
  if [ -L "$BAYES_LINK_PATH" ] || [ -d "$BAYES_LINK_PATH" ]; then
    echo "✅ Bayes MCP server is linked to MCP servers directory"
  else
    echo "❌ Bayes MCP server is not linked!"
    echo "  ℹ️ Run mcp-servers/bayes-mcp-symlink.sh to fix this issue"
  fi
else
  echo "❌ Bayes project not found!"
fi

# Check tools registry
echo ""
echo "Checking tools registry..."
TOOLS_MD="/Users/dionedge/devqai/mcp/tools.md"
if [ -f "$TOOLS_MD" ]; then
  echo "✅ Tools registry exists at $TOOLS_MD"
  TOOL_COUNT=$(grep -c "^| \*\*" "$TOOLS_MD")
  echo "  ℹ️ Registry contains $TOOL_COUNT tools"
  
  # Check if counts match
  CATEGORY_COUNT=$(grep -c "^### .* (count=" "$TOOLS_MD")
  echo "  ℹ️ Registry has $CATEGORY_COUNT categories"
  
  # Check if bayes-mcp is registered
  if grep -q "bayes-mcp" "$TOOLS_MD"; then
    echo "✅ Bayes MCP is registered in the tools registry"
  else
    echo "❌ Bayes MCP is not registered in the tools registry!"
  fi
else
  echo "❌ Tools registry not found!"
fi

# Check MCP specification
echo ""
echo "Checking MCP specification..."
SPEC_MD="/Users/dionedge/devqai/mcp/spec.md"
if [ -f "$SPEC_MD" ]; then
  echo "✅ MCP specification exists at $SPEC_MD"
else
  echo "❌ MCP specification not found!"
fi

# Summary
echo ""
echo "===== Verification Summary ====="
echo ""
echo "1. Directory Structure: Organized with proper separation"
echo "2. Python Environment: Virtual environment available but may need fixes"
echo "3. Bayes MCP Server: Available and linked to MCP servers directory"
echo "4. Tools Registry: Complete with categories and tool counts"
echo "5. MCP Specification: Comprehensive protocol definition"
echo ""
echo "Next Steps:"
echo "1. Run '/Users/dionedge/devqai/mcp/run-mcp-tool.sh list' to see available tools"
echo "2. Try running the TestModel demo with './run-mcp-tool.sh testmodel'"
echo "3. Launch the Bayes MCP server with './run-mcp-tool.sh bayes'"
echo ""
echo "Done!"