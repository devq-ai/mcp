"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  Zap,
  Bot,
  Settings,
  Activity,
  Play,
  Pause,
  RotateCcw,
  Edit,
  Save,
  X,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { formatDuration, formatRelativeTime } from "@/lib/utils";

interface AgentDetail {
  id: string;
  name: string;
  type: "super_agent" | "playbook_agent" | "codifier" | "io" | "custom";
  status: "active" | "idle" | "busy" | "error" | "offline";
  version: string;
  uptime: number;
  started_at: string;
  capabilities: string[];
  current_tasks: number;
  queue_length: number;
  performance_metrics: {
    success_rate: number;
    avg_response_time: number;
    total_executions: number;
    executions_24h: number;
    error_count_24h: number;
    cpu_usage: number;
    memory_usage: number;
    network_io: number;
    disk_usage: number;
  };
  health_score: number;
  last_heartbeat: string;
  configuration: Record<string, any>;
  activity_logs: ActivityLog[];
  performance_history: PerformancePoint[];
  error_logs: ErrorLog[];
}

interface ActivityLog {
  id: string;
  timestamp: string;
  action: string;
  details: string;
  status: "success" | "warning" | "error";
  execution_id?: string;
}

interface PerformancePoint {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  response_time: number;
  success_rate: number;
}

interface ErrorLog {
  id: string;
  timestamp: string;
  error_type: string;
  message: string;
  stack_trace?: string;
  execution_id?: string;
}

export default function AgentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const agentId = params.id as string;

  const [agent, setAgent] = useState<AgentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<
    "overview" | "config" | "logs" | "performance"
  >("overview");
  const [isEditing, setIsEditing] = useState(false);
  const [editedConfig, setEditedConfig] = useState<Record<string, any>>({});

  useEffect(() => {
    // Mock data - replace with actual API call
    const mockAgent: AgentDetail = {
      id: agentId,
      name:
        agentId === "super_agent"
          ? "Super Agent"
          : agentId === "playbook_agent"
            ? "Playbook Agent"
            : agentId === "codifier"
              ? "Codifier Agent"
              : agentId === "io_agent"
                ? "IO Agent"
                : "Unknown Agent",
      type: agentId as any,
      status: "active",
      version: "2.1.0",
      uptime: 86400,
      started_at: new Date(Date.now() - 86400000).toISOString(),
      capabilities:
        agentId === "super_agent"
          ? [
              "orchestration",
              "planning",
              "coordination",
              "chat_interface",
              "decision_making",
            ]
          : agentId === "playbook_agent"
            ? [
                "playbook_creation",
                "step_generation",
                "workflow_design",
                "optimization",
              ]
            : agentId === "codifier"
              ? [
                  "code_generation",
                  "validation",
                  "testing",
                  "optimization",
                  "refactoring",
                ]
              : [
                  "data_ingestion",
                  "file_processing",
                  "api_integration",
                  "data_transformation",
                ],
      current_tasks: 3,
      queue_length: 7,
      performance_metrics: {
        success_rate: 98.5,
        avg_response_time: 245,
        total_executions: 1247,
        executions_24h: 87,
        error_count_24h: 2,
        cpu_usage: 34.2,
        memory_usage: 67.8,
        network_io: 12.4,
        disk_usage: 23.1,
      },
      health_score: 95,
      last_heartbeat: new Date(Date.now() - 30000).toISOString(),
      configuration: {
        max_concurrent_tasks: 10,
        timeout_seconds: 300,
        retry_attempts: 3,
        log_level: "info",
        enable_metrics: true,
        auto_scale: false,
      },
      activity_logs: [
        {
          id: "1",
          timestamp: new Date(Date.now() - 300000).toISOString(),
          action: "Task Completed",
          details: "Successfully processed playbook execution request",
          status: "success",
          execution_id: "exec_001",
        },
        {
          id: "2",
          timestamp: new Date(Date.now() - 600000).toISOString(),
          action: "Configuration Updated",
          details: "Updated max_concurrent_tasks from 8 to 10",
          status: "success",
        },
        {
          id: "3",
          timestamp: new Date(Date.now() - 900000).toISOString(),
          action: "Warning",
          details: "High memory usage detected (85%)",
          status: "warning",
        },
        {
          id: "4",
          timestamp: new Date(Date.now() - 1200000).toISOString(),
          action: "Task Started",
          details: "Began processing complex playbook creation request",
          status: "success",
          execution_id: "exec_002",
        },
      ],
      performance_history: Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() - (23 - i) * 3600000).toISOString(),
        cpu_usage: 30 + Math.random() * 40,
        memory_usage: 50 + Math.random() * 30,
        response_time: 200 + Math.random() * 200,
        success_rate: 95 + Math.random() * 5,
      })),
      error_logs: [
        {
          id: "1",
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          error_type: "TimeoutError",
          message: "Request timeout after 300 seconds",
          execution_id: "exec_003",
        },
        {
          id: "2",
          timestamp: new Date(Date.now() - 7200000).toISOString(),
          error_type: "ValidationError",
          message: "Invalid input parameters for playbook creation",
        },
      ],
    };

    setTimeout(() => {
      setAgent(mockAgent);
      setEditedConfig(mockAgent.configuration);
      setLoading(false);
    }, 1000);

    // Real-time updates
    const interval = setInterval(() => {
      if (mockAgent) {
        setAgent((prev) =>
          prev
            ? {
                ...prev,
                performance_metrics: {
                  ...prev.performance_metrics,
                  cpu_usage: Math.max(
                    5,
                    Math.min(
                      95,
                      prev.performance_metrics.cpu_usage +
                        (Math.random() - 0.5) * 10,
                    ),
                  ),
                  memory_usage: Math.max(
                    10,
                    Math.min(
                      90,
                      prev.performance_metrics.memory_usage +
                        (Math.random() - 0.5) * 5,
                    ),
                  ),
                },
                last_heartbeat: new Date().toISOString(),
              }
            : null,
        );
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [agentId]);

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
        return <Zap className="h-6 w-6 text-neon-magenta" />;
      case "playbook_agent":
        return <Bot className="h-6 w-6 text-neon-lime" />;
      case "codifier":
        return <Settings className="h-6 w-6 text-neon-orange" />;
      case "io_agent":
        return <Network className="h-6 w-6 text-neon-cyan" />;
      default:
        return <Bot className="h-6 w-6 text-muted-foreground" />;
    }
  };

  const handleAgentAction = (action: "start" | "stop" | "restart") => {
    if (!agent) return;

    console.log(`${action} agent:`, agent.id);
    setAgent((prev) =>
      prev
        ? {
            ...prev,
            status: action === "stop" ? "offline" : "active",
            last_heartbeat: new Date().toISOString(),
          }
        : null,
    );
  };

  const handleSaveConfig = () => {
    if (!agent) return;

    setAgent((prev) =>
      prev
        ? {
            ...prev,
            configuration: editedConfig,
          }
        : null,
    );
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    if (agent) {
      setEditedConfig(agent.configuration);
    }
    setIsEditing(false);
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

  if (!agent) {
    return (
      <div className="container mx-auto p-8">
        <div className="text-center py-12">
          <AlertTriangle className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-2xl font-semibold mb-2">Agent Not Found</h2>
          <p className="text-muted-foreground mb-4">
            The agent with ID {agentId} could not be found.
          </p>
          <Button onClick={() => router.push("/agents")}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Agents
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button variant="ghost" onClick={() => router.back()} className="p-2">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex items-center space-x-3">
            {getAgentTypeIcon(agent.type)}
            <div>
              <h1 className="text-4xl font-bold text-gradient">{agent.name}</h1>
              <p className="text-xl text-muted-foreground">
                v{agent.version} • {agent.type.replace("_", " ")}
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Badge className={getStatusColor(agent.status)}>
              {agent.status.toUpperCase()}
            </Badge>
            {getStatusIcon(agent.status)}
          </div>

          {agent.status === "offline" ? (
            <Button
              onClick={() => handleAgentAction("start")}
              className="btn-neon text-neon-green"
            >
              <Play className="h-4 w-4 mr-2" />
              Start
            </Button>
          ) : (
            <div className="flex space-x-2">
              <Button
                variant="outline"
                onClick={() => handleAgentAction("restart")}
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                Restart
              </Button>
              <Button
                variant="outline"
                onClick={() => handleAgentAction("stop")}
                className="text-neon-red border-neon-red hover:bg-neon-red/10"
              >
                <Pause className="h-4 w-4 mr-2" />
                Stop
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Agent Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Health Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-neon-green">
              {agent.health_score}%
            </div>
            <Progress value={agent.health_score} className="mt-2" />
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Uptime</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-lime">
              {formatDuration(agent.uptime)}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Since {formatRelativeTime(agent.started_at)}
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Active Tasks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-orange">
              {agent.current_tasks}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {agent.queue_length} in queue
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Success Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-green">
              {agent.performance_metrics.success_rate.toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {agent.performance_metrics.executions_24h} executions today
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 border-b border-border">
        {[
          { id: "overview", label: "Overview" },
          { id: "config", label: "Configuration" },
          { id: "logs", label: "Activity Logs" },
          { id: "performance", label: "Performance" },
        ].map((tab) => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? "default" : "ghost"}
            onClick={() => setActiveTab(tab.id as any)}
            className={cn(
              "rounded-b-none",
              activeTab === tab.id && "btn-neon text-neon-magenta",
            )}
          >
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "overview" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Performance Metrics */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-muted-foreground">
                      Avg Response Time
                    </div>
                    <div className="text-lg font-semibold">
                      {agent.performance_metrics.avg_response_time}ms
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">
                      Total Executions
                    </div>
                    <div className="text-lg font-semibold">
                      {agent.performance_metrics.total_executions.toLocaleString()}
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
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
                  <div>
                    <div className="flex justify-between text-sm mb-1">
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
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Network I/O</span>
                      <span>
                        {agent.performance_metrics.network_io.toFixed(1)} MB/s
                      </span>
                    </div>
                    <Progress
                      value={agent.performance_metrics.network_io * 5}
                      className="h-2"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Disk Usage</span>
                      <span>
                        {agent.performance_metrics.disk_usage.toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={agent.performance_metrics.disk_usage}
                      className="h-2"
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Capabilities & Status */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Capabilities & Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-muted-foreground mb-2">
                    Capabilities
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {agent.capabilities.map((capability) => (
                      <Badge
                        key={capability}
                        variant="outline"
                        className="text-xs"
                      >
                        {capability}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="text-sm text-muted-foreground mb-2">
                    System Information
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Last Heartbeat</span>
                      <span>{formatRelativeTime(agent.last_heartbeat)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Errors (24h)</span>
                      <span className="text-neon-red">
                        {agent.performance_metrics.error_count_24h}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Started</span>
                      <span>{formatRelativeTime(agent.started_at)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "config" && (
        <Card className="cyber-card">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Configuration</CardTitle>
                <CardDescription>
                  Agent configuration parameters and settings
                </CardDescription>
              </div>
              <div className="flex space-x-2">
                {isEditing ? (
                  <>
                    <Button
                      size="sm"
                      onClick={handleSaveConfig}
                      className="btn-neon text-neon-green"
                    >
                      <Save className="h-4 w-4 mr-2" />
                      Save
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleCancelEdit}
                    >
                      <X className="h-4 w-4 mr-2" />
                      Cancel
                    </Button>
                  </>
                ) : (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setIsEditing(true)}
                  >
                    <Edit className="h-4 w-4 mr-2" />
                    Edit
                  </Button>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(editedConfig).map(([key, value]) => (
                <div key={key} className="space-y-2">
                  <Label htmlFor={key}>
                    {key
                      .replace(/_/g, " ")
                      .replace(/\b\w/g, (l) => l.toUpperCase())}
                  </Label>
                  {isEditing ? (
                    <Input
                      id={key}
                      value={value?.toString() || ""}
                      onChange={(e) =>
                        setEditedConfig((prev) => ({
                          ...prev,
                          [key]:
                            typeof value === "number"
                              ? Number(e.target.value)
                              : e.target.value,
                        }))
                      }
                      type={typeof value === "number" ? "number" : "text"}
                    />
                  ) : (
                    <div className="p-2 bg-muted/20 rounded border">
                      {value?.toString() || "N/A"}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === "logs" && (
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle>Activity Logs</CardTitle>
            <CardDescription>
              Recent agent activities and system events
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {agent.activity_logs.map((log) => (
                <div
                  key={log.id}
                  className="flex items-start space-x-4 p-4 rounded-lg border border-border/50"
                >
                  <div className="flex-shrink-0 mt-1">
                    {log.status === "success" && (
                      <CheckCircle className="h-4 w-4 text-neon-green" />
                    )}
                    {log.status === "warning" && (
                      <AlertTriangle className="h-4 w-4 text-neon-orange" />
                    )}
                    {log.status === "error" && (
                      <XCircle className="h-4 w-4 text-neon-red" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">{log.action}</h4>
                      <span className="text-xs text-muted-foreground">
                        {formatRelativeTime(log.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {log.details}
                    </p>
                    {log.execution_id && (
                      <Badge variant="outline" className="text-xs mt-2">
                        Execution: {log.execution_id}
                      </Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === "performance" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Performance Trends (24h)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center text-muted-foreground">
                  Performance charts will be implemented with Recharts
                </div>
                {/* TODO: Implement charts with Recharts */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Peak CPU</div>
                    <div className="font-medium text-neon-orange">78.5%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Peak Memory</div>
                    <div className="font-medium text-neon-orange">85.2%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Best Response</div>
                    <div className="font-medium text-neon-green">89ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Worst Response</div>
                    <div className="font-medium text-neon-red">1.2s</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Error Logs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {agent.error_logs.map((error) => (
                  <div
                    key={error.id}
                    className="p-3 rounded border border-neon-red/20 bg-neon-red/5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-neon-red">
                        {error.error_type}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {formatRelativeTime(error.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm mt-1">{error.message}</p>
                    {error.execution_id && (
                      <Badge variant="outline" className="text-xs mt-2">
                        {error.execution_id}
                      </Badge>
                    )}
                  </div>
                ))}
                {agent.error_logs.length === 0 && (
                  <div className="text-center text-muted-foreground py-4">
                    No errors in the last 24 hours
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
