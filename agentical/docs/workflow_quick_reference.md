# Workflow Quick Reference Guide

## 🚀 Quick Start: Which Workflow Type Do I Need?

### Use **System Workflows** (`/api/v1/workflows/`) when you need to:
- ✅ Coordinate multiple agents working together
- ✅ Integrate with external systems (databases, APIs, tools)
- ✅ Automate complete business processes
- ✅ Handle long-running operations (hours to days)
- ✅ Manage complex dependencies between steps

### Use **Agent Workflows** (agent configuration) when you need to:
- ✅ Define how an individual agent solves problems
- ✅ Customize agent behavior for specific tasks
- ✅ Optimize tool usage within an agent
- ✅ Implement agent learning patterns
- ✅ Handle agent-specific error recovery

## 📊 System Workflow Examples

### Data Processing Pipeline
```yaml
# Multi-agent data processing workflow
name: "Customer Analytics Pipeline"
steps:
  - agent_type: "data_science_agent"
    task: "ingest_customer_data"
  - agent_type: "data_science_agent" 
    task: "clean_and_validate"
  - agent_type: "code_agent"
    task: "generate_reports"
```

### CI/CD Automation
```yaml
# Software deployment workflow
name: "Automated Deployment"
steps:
  - agent_type: "code_agent"
    task: "run_tests"
  - agent_type: "devops_agent"
    task: "build_docker_image"
  - agent_type: "devops_agent"
    task: "deploy_to_staging"
```

## 🤖 Agent Workflow Examples

### Code Agent Internal Logic
```python
# How the code agent approaches code review
class CodeReviewWorkflow:
    async def execute(self, code):
        # 1. Static analysis
        issues = await self.analyze_code(code)
        # 2. Security scan
        security = await self.security_check(code)
        # 3. Generate review
        return await self.create_review(issues, security)
```

### Data Science Agent Pattern
```python
# How the data science agent handles analysis
class DataAnalysisWorkflow:
    async def execute(self, dataset):
        # 1. Data quality check
        quality = await self.validate_data(dataset)
        # 2. Statistical analysis
        stats = await self.analyze_statistics(dataset)
        # 3. Generate insights
        return await self.create_insights(stats)
```

## 🔄 How They Work Together

```
System Workflow: "Customer Support Automation"
│
├── Step 1: Classify inquiry (NLP Agent)
│   └── Agent Workflow: Text classification → sentiment analysis → category determination
│
├── Step 2: Route to specialist (Routing Logic)
│
└── Step 3: Resolve issue (Support Agent)
    └── Agent Workflow: Knowledge search → solution generation → response formatting
```

## 📡 API Endpoints at a Glance

### System Workflow Management
```http
POST   /api/v1/workflows/                    # Create business process
GET    /api/v1/workflows/                    # List all workflows
POST   /api/v1/workflows/{id}/execute        # Start execution
POST   /api/v1/workflows/{id}/executions/{eid}/control  # Pause/resume/stop
```

### Agent Management (includes internal workflows)
```http
POST   /api/v1/agents/                       # Create/configure agent
PUT    /api/v1/agents/{id}/config            # Update agent behavior
POST   /api/v1/agents/{id}/execute           # Run agent task
```

### Analytics for Both
```http
GET    /api/v1/analytics/workflows/metrics   # System workflow performance
GET    /api/v1/analytics/agents/metrics      # Agent performance
GET    /api/v1/analytics/system/metrics      # Infrastructure metrics
```

## ⚡ Quick Decision Tree

```
Need to coordinate multiple agents? → System Workflow
Need to integrate with external systems? → System Workflow
Need to automate a business process? → System Workflow
Need to customize how one agent works? → Agent Workflow
Need to optimize agent tool usage? → Agent Workflow
Need to implement agent learning? → Agent Workflow
```

## 🔍 Monitoring Differences

### System Workflow Monitoring
- Track execution across multiple agents
- Monitor business process completion
- Measure end-to-end performance
- Resource usage across systems

### Agent Workflow Monitoring
- Track individual agent performance
- Monitor tool usage effectiveness
- Measure agent decision quality
- Learning and adaptation progress

## 🛠️ Development Workflow

### Creating System Workflows
1. Define business process steps
2. Identify required agents and tools
3. Configure dependencies and conditions
4. Test with sample data
5. Deploy and monitor execution

### Configuring Agent Workflows
1. Analyze agent task requirements
2. Define internal logic patterns
3. Configure tool usage sequences
4. Implement error handling
5. Monitor and optimize performance

## 📚 Related Documentation

- **[Complete Workflow Types Guide](./workflow_types_explanation.md)** - Detailed technical comparison
- **[API Implementation Summary](./api_implementation_summary.md)** - Complete endpoint documentation
- **Agent Configuration Guide** - How to customize agent behaviors
- **System Architecture Guide** - Overall platform architecture

## 🎯 Key Takeaways

1. **System Workflows** = Orchestration of multiple components
2. **Agent Workflows** = Internal logic of individual agents
3. **Both are essential** for a complete automation platform
4. **Clear separation** enables scalability and maintainability
5. **Monitoring both levels** provides comprehensive insights

---
*For technical support or questions, refer to the complete documentation or contact the development team.*