"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import Link from "next/link";
import {
  Zap,
  Bot,
  Settings,
  Activity,
  Play,
  Pause,
  RotateCcw,
  Eye,
  BarChart3,
  CheckCircle,
  Clock,
  XCircle,
  Network,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { formatDuration, formatRelativeTime } from "@/lib/utils";

interface Agent {
  id: string;
  name: string;
  type: "super_agent" | "playbook_agent" | "codifier" | "io" | "custom";
  status: "active" | "idle" | "busy" | "error" | "offline";
  version: string;
  uptime: number;
  capabilities: string[];
  current_tasks: number;
  queue_length: number;
  performance_metrics: {
    success_rate: number;
    avg_response_time: number;
    total_executions: number;
    error_count_24h: number;
    cpu_usage: number;
    memory_usage: number;
  };
  last_heartbeat: string;
  configuration: Record<string, any>;
  health_score: number;
}

interface SystemMetrics {
  total_agents: number;
  active_agents: number;
  busy_agents: number;
  total_tasks_today: number;
  avg_system_load: number;
  overall_success_rate: number;
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    total_agents: 0,
    active_agents: 0,
    busy_agents: 0,
    total_tasks_today: 0,
    avg_system_load: 0,
    overall_success_rate: 0,
  });
  const [loading, setLoading] = useState(true);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"overview" | "detailed">("overview");

  useEffect(() => {
    // Mock data - replace with actual API calls
    const mockAgents: Agent[] = [
      {
        id: "super_agent",
        name: "Super Agent",
        type: "super_agent",
        status: "active",
        version: "2.1.0",
        uptime: 86400,
        capabilities: [
          "orchestration",
          "planning",
          "coordination",
          "chat_interface",
        ],
        current_tasks: 3,
        queue_length: 7,
        performance_metrics: {
          success_rate: 98.5,
          avg_response_time: 245,
          total_executions: 1247,
          error_count_24h: 2,
          cpu_usage: 34.2,
          memory_usage: 67.8,
        },
        last_heartbeat: new Date(Date.now() - 30000).toISOString(),
        configuration: {
          max_concurrent_tasks: 10,
          timeout_seconds: 300,
          retry_attempts: 3,
        },
        health_score: 95,
      },
      {
        id: "playbook_agent",
        name: "Playbook Agent",
        type: "playbook_agent",
        status: "busy",
        version: "1.8.2",
        uptime: 72000,
        capabilities: [
          "playbook_creation",
          "step_generation",
          "workflow_design",
        ],
        current_tasks: 2,
        queue_length: 4,
        performance_metrics: {
          success_rate: 94.2,
          avg_response_time: 1840,
          total_executions: 856,
          error_count_24h: 5,
          cpu_usage: 78.5,
          memory_usage: 45.3,
        },
        last_heartbeat: new Date(Date.now() - 15000).toISOString(),
        configuration: {
          max_steps_per_playbook: 50,
          complexity_threshold: 8,
          auto_optimize: true,
        },
        health_score: 88,
      },
      {
        id: "codifier",
        name: "Codifier Agent",
        type: "codifier",
        status: "active",
        version: "3.0.1",
        uptime: 92400,
        capabilities: [
          "code_generation",
          "validation",
          "testing",
          "optimization",
        ],
        current_tasks: 1,
        queue_length: 2,
        performance_metrics: {
          success_rate: 96.8,
          avg_response_time: 892,
          total_executions: 2134,
          error_count_24h: 3,
          cpu_usage: 52.1,
          memory_usage: 38.9,
        },
        last_heartbeat: new Date(Date.now() - 45000).toISOString(),
        configuration: {
          language_support: ["python", "typescript", "sql"],
          code_quality_threshold: 85,
          auto_format: true,
        },
        health_score: 92,
      },
      {
        id: "io_agent",
        name: "IO Agent",
        type: "io",
        status: "idle",
        version: "2.3.4",
        uptime: 68400,
        capabilities: ["data_ingestion", "file_processing", "api_integration"],
        current_tasks: 0,
        queue_length: 1,
        performance_metrics: {
          success_rate: 99.1,
          avg_response_time: 156,
          total_executions: 3421,
          error_count_24h: 1,
          cpu_usage: 12.3,
          memory_usage: 28.7,
        },
        last_heartbeat: new Date(Date.now() - 20000).toISOString(),
        configuration: {
          max_file_size_mb: 100,
          supported_formats: ["json", "csv", "xml", "yaml"],
          timeout_seconds: 60,
        },
        health_score: 97,
      },
    ];

    const mockSystemMetrics: SystemMetrics = {
      total_agents: 4,
      active_agents: 3,
      busy_agents: 1,
      total_tasks_today: 127,
      avg_system_load: 44.3,
      overall_success_rate: 96.9,
    };

    setTimeout(() => {
      setAgents(mockAgents);
      setSystemMetrics(mockSystemMetrics);
      setLoading(false);
    }, 1000);

    // Simulate real-time updates
    const interval = setInterval(() => {
      setAgents((prev) =>
        prev.map((agent) => ({
          ...agent,
          performance_metrics: {
            ...agent.performance_metrics,
            cpu_usage: Math.max(
              5,
              Math.min(
                95,
                agent.performance_metrics.cpu_usage +
                  (Math.random() - 0.5) * 10,
              ),
            ),
            memory_usage: Math.max(
              10,
              Math.min(
                90,
                agent.performance_metrics.memory_usage +
                  (Math.random() - 0.5) * 5,
              ),
            ),
          },
          last_heartbeat: new Date().toISOString(),
        })),
      );
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "status-done";
      case "busy":
        return "status-doing";
      case "idle":
        return "status-todo";
      case "error":
        return "status-tech-debt";
      case "offline":
        return "status-backlog";
      default:
        return "status-backlog";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active":
        return <CheckCircle className="h-4 w-4 text-neon-green" />;
      case "busy":
        return <Activity className="h-4 w-4 text-neon-lime animate-pulse" />;
      case "idle":
        return <Clock className="h-4 w-4 text-muted-foreground" />;
      case "error":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      case "offline":
        return <XCircle className="h-4 w-4 text-muted" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getAgentTypeIcon = (type: string) => {
    switch (type) {
      case "super_agent":
        return <Zap className="h-5 w-5 text-neon-magenta" />;
      case "playbook_agent":
        return <Bot className="h-5 w-5 text-neon-lime" />;
      case "codifier":
        return <Settings className="h-5 w-5 text-neon-orange" />;
      case "io":
        return <Network className="h-5 w-5 text-neon-cyan" />;
      default:
        return <Bot className="h-5 w-5 text-muted-foreground" />;
    }
  };

  const getHealthColor = (score: number) => {
    if (score >= 90) return "text-neon-green";
    if (score >= 75) return "text-neon-lime";
    if (score >= 60) return "text-neon-orange";
    return "text-neon-red";
  };

  const handleAgentAction = (
    agentId: string,
    action: "start" | "stop" | "restart",
  ) => {
    // Implement agent control actions
    console.log(`${action} agent:`, agentId);

    setAgents((prev) =>
      prev.map((agent) =>
        agent.id === agentId
          ? {
              ...agent,
              status: action === "stop" ? "offline" : "active",
              last_heartbeat: new Date().toISOString(),
            }
          : agent,
      ),
    );
  };

  if (loading) {
    return (
      <div className="container mx-auto p-8">
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="loading-spinner w-8 h-8 text-primary"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-gradient">Agent Management</h1>
          <p className="text-xl text-muted-foreground">
            Monitor and control AI agents across the system
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <Button
            variant={viewMode === "overview" ? "default" : "outline"}
            onClick={() => setViewMode("overview")}
            className={viewMode === "overview" ? "btn-neon text-neon-lime" : ""}
          >
            Overview
          </Button>
          <Button
            variant={viewMode === "detailed" ? "default" : "outline"}
            onClick={() => setViewMode("detailed")}
            className={
              viewMode === "detailed" ? "btn-neon text-neon-magenta" : ""
            }
          >
            Detailed
          </Button>
        </div>
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6">
        <Card className="cyber-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-magenta">
              {systemMetrics.total_agents}
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-green">
              {systemMetrics.active_agents}
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Busy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-lime">
              {systemMetrics.busy_agents}
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Tasks Today</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-orange">
              {systemMetrics.total_tasks_today}
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">System Load</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-cyan">
              {systemMetrics.avg_system_load.toFixed(1)}%
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-green">
              {systemMetrics.overall_success_rate.toFixed(1)}%
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agents Grid */}
      <div
        className={cn(
          "grid gap-6",
          viewMode === "overview"
            ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-2"
            : "grid-cols-1",
        )}
      >
        {agents.map((agent) => (
          <Card
            key={agent.id}
            className={cn(
              "cyber-card cursor-pointer transition-all",
              selectedAgent === agent.id && "border-primary shadow-neon",
            )}
            onClick={() =>
              setSelectedAgent(selectedAgent === agent.id ? null : agent.id)
            }
          >
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getAgentTypeIcon(agent.type)}
                  <div>
                    <CardTitle className="text-lg">{agent.name}</CardTitle>
                    <CardDescription>
                      v{agent.version} • Uptime: {formatDuration(agent.uptime)}
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge className={getStatusColor(agent.status)}>
                    {agent.status}
                  </Badge>
                  {getStatusIcon(agent.status)}
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <div className="space-y-4">
                {/* Health Score */}
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Health Score</span>
                  <span
                    className={cn(
                      "text-lg font-bold",
                      getHealthColor(agent.health_score),
                    )}
                  >
                    {agent.health_score}%
                  </span>
                </div>

                {/* Current Activity */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Current Tasks</div>
                    <div className="font-medium text-neon-lime">
                      {agent.current_tasks}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Queue Length</div>
                    <div className="font-medium">{agent.queue_length}</div>
                  </div>
                </div>

                {/* Performance Metrics */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Success Rate</span>
                    <span className="text-neon-green font-medium">
                      {agent.performance_metrics.success_rate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Avg Response</span>
                    <span className="font-medium">
                      {agent.performance_metrics.avg_response_time}ms
                    </span>
                  </div>
                </div>

                {/* Resource Usage */}
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>CPU Usage</span>
                      <span>
                        {agent.performance_metrics.cpu_usage.toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={agent.performance_metrics.cpu_usage}
                      className="h-2"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Memory Usage</span>
                      <span>
                        {agent.performance_metrics.memory_usage.toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={agent.performance_metrics.memory_usage}
                      className="h-2"
                    />
                  </div>
                </div>

                {/* Capabilities */}
                <div>
                  <div className="text-sm text-muted-foreground mb-2">
                    Capabilities
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {agent.capabilities.slice(0, 3).map((capability) => (
                      <Badge
                        key={capability}
                        variant="outline"
                        className="text-xs"
                      >
                        {capability}
                      </Badge>
                    ))}
                    {agent.capabilities.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{agent.capabilities.length - 3} more
                      </Badge>
                    )}
                  </div>
                </div>

                {/* Last Heartbeat */}
                <div className="text-xs text-muted-foreground">
                  Last heartbeat: {formatRelativeTime(agent.last_heartbeat)}
                </div>

                {/* Actions */}
                {selectedAgent === agent.id && (
                  <div className="flex space-x-2 pt-2 border-t border-border/50">
                    <Link href={`/agents/${agent.id}`}>
                      <Button size="sm" variant="outline" className="flex-1">
                        <Eye className="h-4 w-4 mr-2" />
                        Details
                      </Button>
                    </Link>

                    {agent.status === "offline" ? (
                      <Button
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAgentAction(agent.id, "start");
                        }}
                        className="btn-neon text-neon-green"
                      >
                        <Play className="h-4 w-4 mr-2" />
                        Start
                      </Button>
                    ) : (
                      <div className="flex space-x-1">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAgentAction(agent.id, "restart");
                          }}
                        >
                          <RotateCcw className="h-4 w-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAgentAction(agent.id, "stop");
                          }}
                          className="text-neon-red border-neon-red hover:bg-neon-red/10"
                        >
                          <Pause className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle>System Actions</CardTitle>
          <CardDescription>
            Manage all agents and system-wide configurations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Button
              onClick={() => {
                agents.forEach((agent) => {
                  if (agent.status === "offline") {
                    handleAgentAction(agent.id, "start");
                  }
                });
              }}
              className="btn-neon text-neon-green"
            >
              <Play className="h-4 w-4 mr-2" />
              Start All Offline
            </Button>

            <Button
              onClick={() => {
                agents.forEach((agent) => {
                  handleAgentAction(agent.id, "restart");
                });
              }}
              variant="outline"
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Restart All
            </Button>

            <Button variant="outline">
              <Settings className="h-4 w-4 mr-2" />
              Global Configuration
            </Button>

            <Button variant="outline">
              <BarChart3 className="h-4 w-4 mr-2" />
              Performance Report
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
