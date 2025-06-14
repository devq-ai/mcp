#!/bin/bash

# MCP (Model Context Protocol) Development Environment Setup
# This script sets up the Python environment and dependencies for the MCP repository

set -e

echo "🚀 Setting up MCP Development Environment"
echo "=========================================="

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install Python and essential tools
echo "🐍 Installing Python and development tools..."
sudo apt-get install -y python3 python3-pip python3-venv python3-dev build-essential

# Verify Python installation
echo "✅ Python version:"
python3 --version

# Create virtual environment if it doesn't exist
VENV_PATH="$HOME/pydantic_ai_env"
if [ ! -d "$VENV_PATH" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv "$VENV_PATH"
else
    echo "✅ Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment and add to profile
echo "🔄 Configuring virtual environment activation..."
echo "# MCP Virtual Environment Activation" >> "$HOME/.profile"
echo "source $VENV_PATH/bin/activate" >> "$HOME/.profile"

# Activate virtual environment for current session
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install main dependencies
echo "📚 Installing Pydantic AI and dependencies..."
pip install pydantic-ai

# Install additional dependencies that might be needed
echo "📦 Installing additional Python packages..."
pip install pytest pytest-asyncio

# Verify installation
echo "🔍 Verifying installation..."
python -c "import pydantic_ai; print(f'✅ Pydantic AI version: {pydantic_ai.__version__}')"

# Make Python scripts executable
echo "🔧 Making Python scripts executable..."
find claude4/ -name "*.py" -exec chmod +x {} \;

# Create a simple test to verify the environment
echo "🧪 Creating environment verification test..."
cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify the MCP environment is properly configured."""

import sys
import importlib.util

def test_python_version():
    """Test Python version is 3.8+"""
    version = sys.version_info
    assert version.major == 3 and version.minor >= 8, f"Python 3.8+ required, got {version.major}.{version.minor}"
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")

def test_pydantic_ai():
    """Test Pydantic AI is available"""
    try:
        import pydantic_ai
        print(f"✅ Pydantic AI {pydantic_ai.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Pydantic AI import failed: {e}")
        return False

def test_testmodel():
    """Test TestModel is available"""
    try:
        from pydantic_ai.models.test import TestModel
        test_model = TestModel()
        print("✅ TestModel available")
        return True
    except ImportError as e:
        print(f"❌ TestModel import failed: {e}")
        return False

def test_main_script():
    """Test main demo script exists and is importable"""
    import os
    script_path = "claude4/examples/testmodel_demo.py"
    if os.path.exists(script_path):
        print(f"✅ Main demo script exists: {script_path}")
        return True
    else:
        print(f"❌ Main demo script not found: {script_path}")
        return False

if __name__ == "__main__":
    print("🧪 Testing MCP Environment Configuration")
    print("=" * 50)
    
    tests = [
        test_python_version,
        test_pydantic_ai,
        test_testmodel,
        test_main_script
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print(f"\n📊 Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 Environment setup successful!")
        sys.exit(0)
    else:
        print("❌ Environment setup incomplete")
        sys.exit(1)
EOF

chmod +x test_environment.py

echo ""
echo "✅ MCP Development Environment Setup Complete!"
echo ""
echo "📋 Summary:"
echo "  - Python 3.x installed and configured"
echo "  - Virtual environment created at $VENV_PATH"
echo "  - Pydantic AI installed"
echo "  - Virtual environment activation added to ~/.profile"
echo "  - Python scripts made executable"
echo ""
echo "🚀 Ready to run MCP tools and tests!"