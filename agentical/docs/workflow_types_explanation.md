# Workflow Types in Agentical: System Workflows vs Agent Workflows

## Overview

The Agentical platform utilizes two distinct types of workflows that serve different purposes and operate at different levels of the system. This document clarifies the differences, use cases, and interactions between **System Workflows** and **Agent Workflows**.

## System Workflows

### Definition
System Workflows are **orchestration-level workflows** that manage the coordination and execution of multiple agents, tools, and processes across the entire Agentical platform. They represent high-level business processes and automation sequences.

### Characteristics
- **Multi-Agent Orchestration**: Coordinate multiple agents working together
- **Cross-System Integration**: Integrate with external systems, databases, and APIs
- **Business Process Automation**: Represent complete business workflows
- **Stateful Execution**: Maintain state across multiple steps and sessions
- **Long-Running**: Can execute over extended periods (hours, days, weeks)
- **Complex Dependencies**: Support conditional logic, loops, and parallel execution
- **Resource Management**: Manage system resources and execution priorities

### Examples
```yaml
# Data Processing Pipeline Workflow
name: "Customer Data Processing Pipeline"
type: "sequential"
steps:
  - id: "ingest"
    type: "agent_task"
    agent_type: "data_science_agent"
    task: "ingest_customer_data"
    
  - id: "validate"
    type: "agent_task" 
    agent_type: "data_science_agent"
    task: "validate_data_quality"
    dependencies: ["ingest"]
    
  - id: "analyze"
    type: "parallel"
    steps:
      - agent_type: "data_science_agent"
        task: "demographic_analysis"
      - agent_type: "data_science_agent" 
        task: "behavioral_analysis"
    dependencies: ["validate"]
    
  - id: "report"
    type: "agent_task"
    agent_type: "code_agent"
    task: "generate_insights_report"
    dependencies: ["analyze"]
```

### Use Cases
- **ETL Pipelines**: Data extraction, transformation, and loading processes
- **CI/CD Workflows**: Automated deployment and testing pipelines
- **Business Automation**: Invoice processing, customer onboarding, compliance checks
- **Multi-Step Analysis**: Complex research and analysis requiring multiple agents
- **Integration Workflows**: Synchronizing data between multiple systems
- **Monitoring Workflows**: Automated system health checks and maintenance

### Management
- Managed via `/api/v1/workflows/` endpoints
- Stored in the workflows database tables
- Executed by the Workflow Engine
- Monitored through the Analytics & Monitoring system

## Agent Workflows

### Definition
Agent Workflows are **internal execution patterns** that define how individual agents process tasks and make decisions. They represent the cognitive and operational patterns within a single agent's execution context.

### Characteristics
- **Single-Agent Scope**: Execute within the context of one agent
- **Task-Specific Logic**: Define how to approach and solve specific types of problems
- **Cognitive Patterns**: Represent thinking and decision-making processes
- **Tool Orchestration**: Manage the sequence of tool usage within an agent
- **Context-Aware**: Adapt based on current context and available information
- **Short-Running**: Typically complete within minutes or hours
- **Dynamic Adaptation**: Can modify execution based on intermediate results

### Examples
```python
# Code Agent's Code Review Workflow
class CodeReviewWorkflow:
    def __init__(self, agent_context):
        self.context = agent_context
        self.tools = ["static_analyzer", "security_scanner", "test_runner"]
    
    async def execute(self, code_submission):
        # Step 1: Initial code analysis
        analysis = await self.static_analysis(code_submission)
        
        # Step 2: Security review (conditional)
        if analysis.has_security_concerns:
            security_report = await self.security_scan(code_submission)
            analysis.add_security_findings(security_report)
        
        # Step 3: Test coverage check
        test_results = await self.run_tests(code_submission)
        
        # Step 4: Generate comprehensive review
        review = await self.generate_review(analysis, test_results)
        
        return review
```

### Use Cases
- **Problem-Solving Patterns**: How agents approach different types of problems
- **Tool Usage Sequences**: Optimal sequences for using available tools
- **Decision Trees**: Logic for making choices during task execution
- **Error Recovery**: How agents handle and recover from errors
- **Learning Adaptation**: How agents modify their approach based on feedback
- **Context Switching**: How agents adapt to different types of requests

### Management
- Defined in agent class implementations
- Configured through agent configuration systems
- Monitored through agent execution tracking
- Optimized through agent performance analytics

## Key Differences

| Aspect | System Workflows | Agent Workflows |
|--------|------------------|------------------|
| **Scope** | Multi-agent, cross-system | Single agent, internal |
| **Purpose** | Business process orchestration | Cognitive/operational patterns |
| **Duration** | Long-running (hours to days) | Short-running (minutes to hours) |
| **Complexity** | High-level coordination | Task-specific execution |
| **State Management** | Persistent across systems | Agent memory and context |
| **Dependencies** | External systems and agents | Tools and internal logic |
| **Visibility** | Fully visible and manageable | Internal to agent implementation |
| **Modification** | Through workflow management APIs | Through agent configuration |

## Interaction Patterns

### System Workflows Invoking Agents
```yaml
# System workflow step that uses an agent
- id: "code_analysis"
  type: "agent_task"
  agent_type: "code_agent"
  config:
    task: "analyze_codebase"
    repository_url: "https://github.com/user/repo"
    analysis_depth: "comprehensive"
  # The agent will use its internal workflow to complete this task
```

### Agent Workflows in System Context
```python
class DataScienceAgent:
    async def execute_task(self, task_request):
        # Agent receives task from system workflow
        if task_request.type == "data_analysis":
            # Agent uses its internal workflow
            result = await self.data_analysis_workflow.execute(
                dataset=task_request.dataset,
                analysis_type=task_request.analysis_type
            )
            # Result is returned to system workflow
            return result
```

## Architecture Integration

### System Workflow Layer
```
┌─────────────────────────────────────────────────────────────┐
│                   System Workflows                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Workflow  │  │   Workflow  │  │   Workflow  │       │
│  │   Engine    │  │   Manager   │  │   Registry  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │    Agent    │  │    Agent    │  │    Agent    │       │
│  │ (Internal   │  │ (Internal   │  │ (Internal   │       │
│  │ Workflows)  │  │ Workflows)  │  │ Workflows)  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## API Distinction

### System Workflow APIs
```http
# Managing system workflows
POST   /api/v1/workflows/                    # Create workflow
GET    /api/v1/workflows/                    # List workflows
PUT    /api/v1/workflows/{id}                # Update workflow
DELETE /api/v1/workflows/{id}                # Delete workflow

# Executing system workflows
POST   /api/v1/workflows/{id}/execute        # Start execution
POST   /api/v1/workflows/{id}/executions/{eid}/control  # Control execution
```

### Agent Workflow APIs
```http
# Agent management (includes internal workflow configuration)
POST   /api/v1/agents/                       # Create/configure agent
GET    /api/v1/agents/{id}/config            # Get agent configuration
PUT    /api/v1/agents/{id}/config            # Update agent configuration

# Agent execution (triggers internal workflows)
POST   /api/v1/agents/{id}/execute           # Execute agent task
GET    /api/v1/agents/{id}/executions        # Get execution history
```

## Monitoring and Analytics

### System Workflow Monitoring
- **Workflow Execution Metrics**: Success rates, duration, throughput
- **Step-by-Step Tracking**: Individual step performance and status
- **Resource Utilization**: System resource usage during execution
- **Cross-Agent Coordination**: Communication and handoff efficiency

### Agent Workflow Monitoring  
- **Agent Performance**: Response times, accuracy, efficiency
- **Tool Usage Patterns**: Which tools are used and how effectively
- **Decision Quality**: Effectiveness of agent decision-making
- **Learning Progress**: How agents improve over time

## Best Practices

### When to Use System Workflows
- ✅ Coordinating multiple agents for complex tasks
- ✅ Integrating with external systems and APIs
- ✅ Implementing business processes with multiple steps
- ✅ Managing long-running automation sequences
- ✅ Handling complex dependencies and conditional logic

### When to Use Agent Workflows
- ✅ Defining how agents approach specific problem types
- ✅ Optimizing tool usage sequences within agents
- ✅ Implementing agent learning and adaptation patterns
- ✅ Managing agent error handling and recovery
- ✅ Customizing agent behavior for specific contexts

### Integration Guidelines
1. **Clear Separation**: Keep system orchestration separate from agent implementation
2. **Standard Interfaces**: Use consistent APIs for system-agent communication
3. **Context Passing**: Ensure proper context and state transfer between levels
4. **Error Propagation**: Handle errors appropriately at each level
5. **Monitoring**: Implement monitoring at both system and agent levels

## Example: Complete Integration

### System Workflow: Customer Support Automation
```yaml
name: "Automated Customer Support"
description: "Handle customer inquiries with automated triage and response"
type: "conditional"
steps:
  - id: "classify"
    type: "agent_task"
    agent_type: "nlp_agent"
    task: "classify_inquiry"
    # Agent uses its internal classification workflow
    
  - id: "route"
    type: "condition"
    condition: "classification.category"
    branches:
      "technical":
        - agent_type: "tech_support_agent"
          task: "resolve_technical_issue"
          # Agent uses its internal troubleshooting workflow
      "billing":
        - agent_type: "billing_agent" 
          task: "handle_billing_inquiry"
          # Agent uses its internal billing workflow
      "general":
        - agent_type: "general_support_agent"
          task: "provide_general_assistance"
          # Agent uses its internal assistance workflow
```

### Agent Workflow: Tech Support Agent Internal Logic
```python
class TechSupportWorkflow:
    async def resolve_technical_issue(self, inquiry):
        # Step 1: Analyze the problem
        problem_analysis = await self.analyze_problem(inquiry)
        
        # Step 2: Search knowledge base
        solutions = await self.search_solutions(problem_analysis)
        
        # Step 3: If no solution found, escalate
        if not solutions:
            return await self.escalate_to_human(inquiry)
        
        # Step 4: Provide solution
        response = await self.format_solution(solutions[0])
        
        # Step 5: Follow up if needed
        if problem_analysis.complexity > 7:
            await self.schedule_followup(inquiry.customer_id)
        
        return response
```

## Conclusion

Understanding the distinction between System Workflows and Agent Workflows is crucial for effective use of the Agentical platform:

- **System Workflows** handle the "what" and "when" - orchestrating high-level business processes
- **Agent Workflows** handle the "how" - defining the internal logic and patterns agents use to complete tasks

Both types work together to create a comprehensive automation platform that can handle complex, multi-step processes while maintaining flexibility and efficiency at the individual agent level.

This separation of concerns allows for:
- **Scalability**: System workflows can orchestrate hundreds of agents
- **Maintainability**: Agent logic can be updated independently
- **Reusability**: Agents with proven workflows can be used across multiple system workflows
- **Optimization**: Each level can be optimized for its specific concerns
- **Clarity**: Clear boundaries make the system easier to understand and manage