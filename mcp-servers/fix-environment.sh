#!/bin/bash
# Fix Python environment and MCP tool configuration

# Define paths
VENV_PATH="/Users/dionedge/devqai/mcp/pydantic_ai_env"
PYTHON_PATH="/opt/homebrew/opt/python@3.13/bin/python3.13"
PYTHON_BACKUP_PATH="/opt/homebrew/bin/python3"

echo "MCP Environment Repair Tool"
echo "=========================="

# Check if Python 3.13 exists
if [ ! -f "$PYTHON_PATH" ]; then
  echo "❌ Python 3.13 not found at $PYTHON_PATH"
  
  if [ -f "$PYTHON_BACKUP_PATH" ]; then
    echo "ℹ️ Using backup Python at $PYTHON_BACKUP_PATH"
    PYTHON_PATH="$PYTHON_BACKUP_PATH"
  else
    echo "❌ Backup Python not found either. Please install Python 3.13 or update paths."
    exit 1
  fi
fi

echo "✅ Using Python at: $PYTHON_PATH"

# Fix virtual environment
if [ -d "$VENV_PATH" ]; then
  echo "ℹ️ Fixing virtual environment at $VENV_PATH"
  
  # Fix activation script
  cat > "$VENV_PATH/bin/activate" << EOL
# This file must be used with "source bin/activate" *from bash*
# you cannot run it directly

deactivate () {
    # reset old environment variables
    if [ -n "\${_OLD_VIRTUAL_PATH:-}" ] ; then
        PATH="\$_OLD_VIRTUAL_PATH"
        export PATH
        unset _OLD_VIRTUAL_PATH
    fi
    if [ -n "\${_OLD_VIRTUAL_PYTHONHOME:-}" ] ; then
        PYTHONHOME="\$_OLD_VIRTUAL_PYTHONHOME"
        export PYTHONHOME
        unset _OLD_VIRTUAL_PYTHONHOME
    fi

    # This should detect bash and zsh, which have a hash command that must
    # be called to get it to forget past commands.  Without forgetting
    # past commands the \$PATH changes we made may not be respected
    if [ -n "\${BASH:-}" -o -n "\${ZSH_VERSION:-}" ] ; then
        hash -r 2> /dev/null
    fi

    if [ -n "\${_OLD_VIRTUAL_PS1:-}" ] ; then
        PS1="\$_OLD_VIRTUAL_PS1"
        export PS1
        unset _OLD_VIRTUAL_PS1
    fi

    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT
    if [ ! "\${1:-}" = "nondestructive" ] ; then
    # Self destruct!
        unset -f deactivate
    fi
}

# unset irrelevant variables
deactivate nondestructive

VIRTUAL_ENV="$VENV_PATH"
export VIRTUAL_ENV

_OLD_VIRTUAL_PATH="\$PATH"
PATH="\$VIRTUAL_ENV/bin:\$PATH"
export PATH

if [ -z "\${VIRTUAL_ENV_DISABLE_PROMPT:-}" ] ; then
    _OLD_VIRTUAL_PS1="\${PS1:-}"
    PS1="\$(basename "\$VIRTUAL_ENV") \${PS1:-}"
    export PS1
    VIRTUAL_ENV_PROMPT="\$(basename "\$VIRTUAL_ENV") "
    export VIRTUAL_ENV_PROMPT
fi

# This variable defines whether or not to create the Python symlinks
RECREATE_SYMLINKS="yes"
if [ "\$RECREATE_SYMLINKS" = "yes" ]; then
  # Create the symlinks, overwriting any existing ones
  if [ -f "\$PYTHON_PATH" ]; then
    ln -sf $PYTHON_PATH "\$VIRTUAL_ENV/bin/python3.13"
    ln -sf "\$VIRTUAL_ENV/bin/python3.13" "\$VIRTUAL_ENV/bin/python3"
    ln -sf "\$VIRTUAL_ENV/bin/python3.13" "\$VIRTUAL_ENV/bin/python"
    echo "✅ Fixed Python symlinks in virtual environment"
  fi
fi

# This should detect bash and zsh, which have a hash command that must
# be called to get it to forget past commands.  Without forgetting
# past commands the \$PATH changes we made may not be respected
if [ -n "\${BASH:-}" -o -n "\${ZSH_VERSION:-}" ] ; then
    hash -r 2> /dev/null
fi
EOL

  echo "✅ Fixed activation script"
  
  # Create environment variables file for MCP
  if [ ! -f "$VENV_PATH/mcp.env" ]; then
    cat > "$VENV_PATH/mcp.env" << EOL
# MCP Environment Variables
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8080
MCP_REGISTRY_URL=https://registry.mcp-protocol.org
MCP_DEBUG=true
MCP_LOG_LEVEL=info
EOL
    echo "✅ Created MCP environment variables file"
  fi
  
else
  echo "❌ Virtual environment not found at $VENV_PATH"
  exit 1
fi

# Create a convenience script to run MCP tools
cat > "/Users/dionedge/devqai/mcp/run-mcp-tool.sh" << EOL
#!/bin/bash
# Run an MCP tool with proper environment setup

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Load environment variables
if [ -f "$VENV_PATH/mcp.env" ]; then
  set -o allexport
  source "$VENV_PATH/mcp.env"
  set +o allexport
fi

# Run the specified MCP tool
if [ "\$1" = "testmodel" ]; then
  echo "Running TestModel demo..."
  python "$VENV_PATH/../claude4/examples/testmodel_demo.py"
elif [ "\$1" = "bayes" ]; then
  echo "Starting Bayes MCP server..."
  cd "/Users/dionedge/devqai/bayes"
  python bayes_mcp.py
elif [ "\$1" = "list" ]; then
  echo "Available MCP tools:"
  echo "  testmodel - Run the TestModel demo"
  echo "  bayes     - Start the Bayes MCP server"
  echo "  help      - Show this help message"
else
  echo "Usage: \$0 [tool]"
  echo ""
  echo "Available tools:"
  echo "  testmodel - Run the TestModel demo"
  echo "  bayes     - Start the Bayes MCP server"
  echo "  list      - List all available tools"
fi
EOL

chmod +x "/Users/dionedge/devqai/mcp/run-mcp-tool.sh"
echo "✅ Created run-mcp-tool.sh script"

echo ""
echo "✅ Environment repair complete!"
echo ""
echo "To activate the environment and run tools:"
echo "  1. source $VENV_PATH/bin/activate"
echo "  2. ./run-mcp-tool.sh [testmodel|bayes|list]"
echo ""