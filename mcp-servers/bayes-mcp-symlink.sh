#!/bin/bash
# Create symbolic link to the bayes project as an MCP server

# Define paths
SOURCE_PATH="/Users/dionedge/devqai/bayes"
TARGET_PATH="/Users/dionedge/devqai/mcp/mcp-servers/bayes-mcp"

# Check if source exists
if [ ! -d "$SOURCE_PATH" ]; then
  echo "Error: Source directory $SOURCE_PATH does not exist!"
  exit 1
fi

# Create symbolic link
ln -sf "$SOURCE_PATH" "$TARGET_PATH"

# Verify the link was created
if [ -L "$TARGET_PATH" ]; then
  echo "✅ Successfully linked $SOURCE_PATH to $TARGET_PATH"
else
  echo "❌ Failed to create symbolic link!"
  exit 1
fi

# Display server info
echo "Bayes MCP Server Information:"
echo "----------------------------"
echo "Source: $SOURCE_PATH"
echo "Link: $TARGET_PATH"
if [ -f "$SOURCE_PATH/bayes_mcp.py" ]; then
  echo "Entry point: bayes_mcp.py"
else
  echo "Warning: Entry point bayes_mcp.py not found!"
fi

echo "Done!"