# Darwin MCP Server

An advanced Model Context Protocol (MCP) server that provides genetic algorithm optimization capabilities to AI agents and applications. Built on the Darwin optimization platform, this server enables AI systems to solve complex optimization problems using evolutionary computation.

## 🧬 Overview

Darwin MCP Server exposes a comprehensive genetic algorithm toolkit through the Model Context Protocol, allowing AI agents to:

- Create and configure optimization problems
- Run single and multi-objective optimizations
- Monitor optimization progress in real-time
- Analyze and visualize optimization results
- Access pre-built optimization templates
- Compare algorithm performance

## 🚀 Features

### Core Optimization Capabilities
- **Single-Objective Optimization**: Traditional genetic algorithms for single-goal problems
- **Multi-Objective Optimization**: NSGA-II and NSGA-III for Pareto-optimal solutions
- **Constraint Handling**: Penalty methods and repair operators for constrained problems
- **Mixed Variable Types**: Support for continuous, discrete, categorical, and binary variables

### AI Agent Integration
- **MCP Protocol Compliance**: Full compatibility with MCP clients and tools
- **Real-time Progress**: Live optimization monitoring and progress updates
- **Template System**: Pre-configured templates for common optimization scenarios
- **Problem Analysis**: Automatic complexity analysis and algorithm recommendations

### Advanced Features
- **Parallel Execution**: Multi-core optimization for improved performance
- **Adaptive Parameters**: Self-adjusting algorithm parameters during evolution
- **Result Visualization**: Interactive charts and graphs for result analysis
- **Performance Metrics**: Comprehensive optimization performance tracking

## 📦 Installation

### From PyPI
```bash
pip install darwin-mcp
```

### From Source
```bash
git clone https://github.com/devqai/darwin.git
cd darwin/mcp/mcp-servers/darwin-mcp
pip install -e .
```

### Using Poetry
```bash
cd darwin/mcp/mcp-servers/darwin-mcp
poetry install
```

## 🎯 Quick Start

### Starting the MCP Server
```bash
# Start with stdio transport (default)
darwin-mcp

# Or explicitly specify stdio
darwin-mcp serve --stdio

# Check version
darwin-mcp version

# Health check
darwin-mcp health
```

### Using with MCP Clients

#### Python MCP Client
```python
import asyncio
from mcp_client import MCPClient

async def optimize_portfolio():
    # Connect to Darwin MCP server
    client = MCPClient("stdio", command=["darwin-mcp"])
    
    # Create optimization problem
    result = await client.call_tool("create_optimization", {
        "name": "Portfolio Optimization",
        "variables": [
            {"name": "stocks", "type": "continuous", "bounds": [0, 0.8]},
            {"name": "bonds", "type": "continuous", "bounds": [0, 0.5]},
            {"name": "cash", "type": "continuous", "bounds": [0, 0.3]}
        ],
        "objectives": [
            {"name": "maximize_return", "type": "maximize"},
            {"name": "minimize_risk", "type": "minimize"}
        ],
        "constraints": [
            {"type": "equality", "expression": "stocks + bonds + cash == 1.0"}
        ],
        "config": {
            "algorithm": "nsga2",
            "population_size": 100,
            "max_generations": 200
        }
    })
    
    optimization_id = result["optimization_id"]
    
    # Run optimization
    await client.call_tool("run_optimization", {
        "optimization_id": optimization_id,
        "async_mode": True
    })
    
    # Monitor progress
    while True:
        status = await client.call_tool("get_optimization_status", {
            "optimization_id": optimization_id
        })
        
        if status["status"] in ["completed", "failed"]:
            break
        
        print(f"Progress: {status.get('progress', 0)*100:.1f}%")
        await asyncio.sleep(2)
    
    # Get results
    results = await client.call_tool("get_optimization_results", {
        "optimization_id": optimization_id,
        "include_visualization": True
    })
    
    return results

# Run the optimization
results = asyncio.run(optimize_portfolio())
print(f"Best solutions found: {len(results['pareto_frontier'])}")
```

#### Claude Desktop Integration
Add to your Claude Desktop MCP configuration:

```json
{
  "mcp_servers": {
    "darwin-mcp": {
      "command": "darwin-mcp",
      "args": ["serve", "--stdio"],
      "description": "Genetic algorithm optimization server"
    }
  }
}
```

## 🛠️ Available Tools

### Core Optimization Tools

#### `create_optimization`
Create a new genetic algorithm optimization problem.

```python
await client.call_tool("create_optimization", {
    "name": "Function Optimization",
    "variables": [
        {"name": "x", "type": "continuous", "bounds": [-5, 5]},
        {"name": "y", "type": "continuous", "bounds": [-5, 5]}
    ],
    "objectives": [
        {"name": "minimize", "type": "minimize", "function": "rastrigin"}
    ],
    "config": {
        "population_size": 50,
        "max_generations": 100
    }
})
```

#### `run_optimization`
Execute an optimization problem.

```python
await client.call_tool("run_optimization", {
    "optimization_id": "opt_123",
    "async_mode": False  # Set to True for background execution
})
```

#### `get_optimization_status`
Check the current status of an optimization.

```python
await client.call_tool("get_optimization_status", {
    "optimization_id": "opt_123"
})
```

#### `get_optimization_results`
Retrieve results from a completed optimization.

```python
await client.call_tool("get_optimization_results", {
    "optimization_id": "opt_123",
    "include_history": True,
    "include_visualization": True
})
```

### Template and Analysis Tools

#### `create_template`
Generate optimization templates for common problems.

```python
await client.call_tool("create_template", {
    "template_type": "portfolio_optimization",
    "parameters": {
        "num_assets": 5,
        "risk_tolerance": 0.1
    }
})
```

#### `analyze_problem`
Analyze problem complexity and get recommendations.

```python
await client.call_tool("analyze_problem", {
    "variables": [...],
    "objectives": [...],
    "constraints": [...]
})
```

### Visualization and Comparison Tools

#### `visualize_results`
Create charts and visualizations of optimization results.

```python
await client.call_tool("visualize_results", {
    "optimization_id": "opt_123",
    "chart_type": "convergence",  # or "pareto_frontier", "diversity"
    "format": "png"
})
```

#### `compare_algorithms`
Compare performance of different genetic algorithms.

```python
await client.call_tool("compare_algorithms", {
    "problem": {...},
    "algorithms": ["genetic", "nsga2", "differential_evolution"],
    "runs": 5
})
```

### System Tools

#### `list_optimizations`
List all optimization problems.

```python
await client.call_tool("list_optimizations", {
    "status": "completed",
    "limit": 20
})
```

#### `get_system_status`
Get Darwin system status and health metrics.

```python
await client.call_tool("get_system_status", {
    "include_metrics": True
})
```

## 📚 Examples

### Example 1: Function Optimization
```python
# Minimize the Rastrigin function
result = await client.call_tool("create_optimization", {
    "name": "Rastrigin Minimization",
    "variables": [
        {"name": "x1", "type": "continuous", "bounds": [-5.12, 5.12]},
        {"name": "x2", "type": "continuous", "bounds": [-5.12, 5.12]}
    ],
    "objectives": [
        {"name": "minimize_rastrigin", "type": "minimize", "function": "rastrigin"}
    ]
})
```

### Example 2: Multi-Objective Portfolio
```python
# Portfolio optimization with risk-return trade-off
result = await client.call_tool("create_optimization", {
    "name": "Risk-Return Portfolio",
    "variables": [
        {"name": "tech_stocks", "type": "continuous", "bounds": [0, 0.5]},
        {"name": "utilities", "type": "continuous", "bounds": [0, 0.4]},
        {"name": "bonds", "type": "continuous", "bounds": [0, 0.6]}
    ],
    "objectives": [
        {"name": "maximize_return", "type": "maximize"},
        {"name": "minimize_risk", "type": "minimize"}
    ],
    "constraints": [
        {"type": "equality", "expression": "tech_stocks + utilities + bonds == 1.0"}
    ],
    "config": {"algorithm": "nsga2"}
})
```

### Example 3: Neural Network Hyperparameters
```python
# Optimize neural network hyperparameters
template = await client.call_tool("create_template", {
    "template_type": "neural_network_tuning",
    "parameters": {
        "model_type": "feedforward",
        "max_layers": 5
    }
})
```

## 🎯 Problem Templates

Darwin MCP provides built-in templates for common optimization scenarios:

- **`function_optimization`**: Mathematical function optimization
- **`portfolio_optimization`**: Financial portfolio optimization  
- **`neural_network_tuning`**: Neural network hyperparameter optimization
- **`scheduling`**: Task scheduling and resource allocation
- **`design_optimization`**: Engineering design optimization

Each template includes pre-configured variables, objectives, constraints, and algorithm settings optimized for the specific problem type.

## ⚙️ Configuration

### Environment Variables
```bash
# MCP Server Configuration
DARWIN_MCP_HOST=localhost
DARWIN_MCP_PORT=8000
DARWIN_MCP_LOG_LEVEL=INFO

# Algorithm Configuration  
DARWIN_DEFAULT_POPULATION_SIZE=100
DARWIN_DEFAULT_MAX_GENERATIONS=200
DARWIN_ENABLE_PARALLEL=true

# Performance Settings
DARWIN_MAX_CONCURRENT_OPTIMIZATIONS=5
DARWIN_MEMORY_LIMIT_GB=4
```

### Configuration File
Create `.darwin-mcp.yaml`:

```yaml
server:
  host: localhost
  port: 8000
  log_level: INFO

algorithms:
  default_population_size: 100
  default_max_generations: 200
  enable_parallel: true
  max_concurrent: 5

performance:
  memory_limit_gb: 4
  cpu_cores: -1  # Use all available cores
```

## 📊 Performance

### Benchmarks
Darwin MCP has been tested on standard optimization problems:

| Problem | Dimensions | Algorithm | Success Rate | Avg. Time |
|---------|------------|-----------|--------------|-----------|
| Sphere | 10 | Genetic | 95% | 2.3s |
| Rastrigin | 10 | Genetic | 87% | 8.1s |
| DTLZ2 | 10 (3 obj) | NSGA-II | 92% | 15.4s |
| Portfolio | 20 assets | NSGA-II | 89% | 12.7s |

### Performance Tips
- Use parallel execution for problems with >10 variables
- Enable adaptive parameters for complex problems
- Use appropriate population sizes (50-200 for most problems)
- Consider early stopping for faster convergence

## 🔍 Troubleshooting

### Common Issues

#### MCP Connection Problems
```bash
# Check if server is running
darwin-mcp health

# Test with stdio
echo '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' | darwin-mcp
```

#### Optimization Failures
```python
# Check problem definition
validation_errors = await client.call_tool("analyze_problem", {
    "variables": [...],
    "objectives": [...]
})

# Monitor system status
status = await client.call_tool("get_system_status")
```

#### Performance Issues
- Reduce population size for faster execution
- Enable parallel processing: `"parallel_execution": true`
- Use simpler algorithms for quick results
- Check system resources and memory usage

### Debug Mode
```bash
# Start server in debug mode
DARWIN_MCP_LOG_LEVEL=DEBUG darwin-mcp serve --stdio
```

## 🤝 Contributing

We welcome contributions to Darwin MCP! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/devqai/darwin.git
cd darwin/mcp/mcp-servers/darwin-mcp
poetry install --with dev
pre-commit install
```

### Running Tests
```bash
pytest tests/
pytest tests/ --cov=darwin_mcp --cov-report=html
```

### Code Quality
```bash
black darwin_mcp/
isort darwin_mcp/
mypy darwin_mcp/
ruff check darwin_mcp/
```

## 📄 License

Darwin MCP Server is released under the BSD 3-Clause License. See [LICENSE](../../../LICENSE) for details.

## 🔗 Related Projects

- **[Darwin Platform](../../../)** - The main Darwin genetic algorithm platform
- **[Bayes MCP](../bayes-mcp)** - Bayesian inference MCP server
- **[Context7 MCP](../context7-mcp)** - Advanced context management
- **[PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython)** - Underlying genetic algorithm library

## 📞 Support

- **Documentation**: [Darwin Docs](https://darwin.devq.ai/docs)
- **GitHub Issues**: [Report Issues](https://github.com/devqai/darwin/issues)
- **Discord**: [Join Community](https://discord.gg/devqai)
- **Email**: team@devq.ai

---

**Darwin MCP Server** - Evolving AI Optimization Solutions  
Copyright © 2025 DevQ.ai - All Rights Reserved