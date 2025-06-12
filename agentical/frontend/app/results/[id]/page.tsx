"use client";

import * as React from "react";
import { useState, useEffect, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Play,
  Square,
  CheckCircle2,
  AlertCircle,
  Clock,
  Download,
  Maximize2,
  Minimize2,
  ArrowLeft,
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
import { WorkflowVisualization } from "@/components/playbook/WorkflowVisualization";

interface ExecutionStep {
  id: string;
  name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  started_at?: string;
  completed_at?: string;
  duration?: number;
  agent_id: string;
  agent_name: string;
  progress: number;
}

interface ExecutionLog {
  id: string;
  timestamp: string;
  level: "info" | "warning" | "error" | "debug";
  message: string;
  step_id?: string;
  step_name?: string;
  agent_name?: string;
  metadata?: Record<string, any>;
}

interface PlaybookExecution {
  id: string;
  playbook_id: string;
  playbook_name: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress_percentage: number;
  current_step?: string;
  current_step_name?: string;
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  steps: ExecutionStep[];
  agents: Record<string, { status: "active" | "idle" | "error"; name: string }>;
  input_variables: Record<string, any>;
  output_data?: any;
  error_message?: string;
}

export default function ExecutionResultsPage() {
  const params = useParams();
  const router = useRouter();
  const executionId = params.id as string;

  const [execution, setExecution] = useState<PlaybookExecution | null>(null);
  const [logs, setLogs] = useState<ExecutionLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [logsExpanded, setLogsExpanded] = useState(false);
  const [filterLevel, setFilterLevel] = useState<string>("all");

  const logsEndRef = useRef<HTMLDivElement>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Mock data - replace with actual API call and WebSocket connection
    const mockExecution: PlaybookExecution = {
      id: executionId,
      playbook_id: "pb_001",
      playbook_name: "Data Processing Pipeline",
      status: "running",
      progress_percentage: 65,
      current_step: "step_3",
      current_step_name: "Data Transformation",
      started_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
      steps: [
        {
          id: "step_1",
          name: "Data Ingestion",
          status: "completed",
          started_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
          completed_at: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
          duration: 45,
          agent_id: "io_agent",
          agent_name: "IO Agent",
          progress: 100,
        },
        {
          id: "step_2",
          name: "Data Validation",
          status: "completed",
          started_at: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
          completed_at: new Date(Date.now() - 3 * 60 * 1000).toISOString(),
          duration: 32,
          agent_id: "codifier",
          agent_name: "Codifier Agent",
          progress: 100,
        },
        {
          id: "step_3",
          name: "Data Transformation",
          status: "running",
          started_at: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
          agent_id: "super_agent",
          agent_name: "Super Agent",
          progress: 75,
        },
        {
          id: "step_4",
          name: "Quality Checks",
          status: "pending",
          agent_id: "codifier",
          agent_name: "Codifier Agent",
          progress: 0,
        },
        {
          id: "step_5",
          name: "Output Generation",
          status: "pending",
          agent_id: "io_agent",
          agent_name: "IO Agent",
          progress: 0,
        },
      ],
      agents: {
        super_agent: { status: "active", name: "Super Agent" },
        codifier: { status: "idle", name: "Codifier Agent" },
        io_agent: { status: "idle", name: "IO Agent" },
        playbook_agent: { status: "idle", name: "Playbook Agent" },
      },
      input_variables: {
        source_file: "data.csv",
        target_format: "json",
        validation_rules: ["not_null", "data_type_check"],
      },
    };

    const mockLogs: ExecutionLog[] = [
      {
        id: "1",
        timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
        level: "info",
        message: "Execution started for Data Processing Pipeline",
        metadata: { execution_id: executionId },
      },
      {
        id: "2",
        timestamp: new Date(Date.now() - 4.5 * 60 * 1000).toISOString(),
        level: "info",
        message: "Starting data ingestion from source file: data.csv",
        step_id: "step_1",
        step_name: "Data Ingestion",
        agent_name: "IO Agent",
      },
      {
        id: "3",
        timestamp: new Date(Date.now() - 4.3 * 60 * 1000).toISOString(),
        level: "info",
        message: "Successfully loaded 1,250 records from source",
        step_id: "step_1",
        step_name: "Data Ingestion",
        agent_name: "IO Agent",
      },
      {
        id: "4",
        timestamp: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
        level: "info",
        message: "Data ingestion completed successfully",
        step_id: "step_1",
        step_name: "Data Ingestion",
        agent_name: "IO Agent",
      },
      {
        id: "5",
        timestamp: new Date(Date.now() - 3.8 * 60 * 1000).toISOString(),
        level: "info",
        message:
          "Starting data validation with rules: not_null, data_type_check",
        step_id: "step_2",
        step_name: "Data Validation",
        agent_name: "Codifier Agent",
      },
      {
        id: "6",
        timestamp: new Date(Date.now() - 3.5 * 60 * 1000).toISOString(),
        level: "warning",
        message: "Found 12 records with null values in optional fields",
        step_id: "step_2",
        step_name: "Data Validation",
        agent_name: "Codifier Agent",
      },
      {
        id: "7",
        timestamp: new Date(Date.now() - 3 * 60 * 1000).toISOString(),
        level: "info",
        message: "Data validation completed - 1,238 valid records",
        step_id: "step_2",
        step_name: "Data Validation",
        agent_name: "Codifier Agent",
      },
      {
        id: "8",
        timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
        level: "info",
        message: "Starting data transformation to JSON format",
        step_id: "step_3",
        step_name: "Data Transformation",
        agent_name: "Super Agent",
      },
      {
        id: "9",
        timestamp: new Date(Date.now() - 1.5 * 60 * 1000).toISOString(),
        level: "info",
        message: "Processed 930 records (75% complete)",
        step_id: "step_3",
        step_name: "Data Transformation",
        agent_name: "Super Agent",
      },
    ];

    setTimeout(() => {
      setExecution(mockExecution);
      setLogs(mockLogs);
      setLoading(false);
    }, 1000);

    // Simulate real-time updates
    const interval = setInterval(() => {
      if (mockExecution.status === "running") {
        const newLog: ExecutionLog = {
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          level: Math.random() > 0.8 ? "warning" : "info",
          message: `Processing batch ${Math.floor(Math.random() * 100) + 900}...`,
          step_id: "step_3",
          step_name: "Data Transformation",
          agent_name: "Super Agent",
        };

        setLogs((prev) => [...prev, newLog]);

        // Update progress
        setExecution((prev) =>
          prev
            ? {
                ...prev,
                progress_percentage: Math.min(
                  100,
                  prev.progress_percentage + Math.random() * 5,
                ),
              }
            : null,
        );
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [executionId]);

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-neon-green" />;
      case "running":
        return <Play className="h-4 w-4 text-neon-lime animate-pulse" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-neon-red" />;
      case "pending":
        return <Clock className="h-4 w-4 text-muted-foreground" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "status-done";
      case "running":
        return "status-doing";
      case "failed":
        return "status-tech-debt";
      case "pending":
        return "status-todo";
      case "cancelled":
        return "status-backlog";
      default:
        return "status-backlog";
    }
  };

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case "error":
        return "text-neon-red";
      case "warning":
        return "text-neon-orange";
      case "info":
        return "text-foreground";
      case "debug":
        return "text-muted-foreground";
      default:
        return "text-foreground";
    }
  };

  const getAgentStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-neon-green";
      case "idle":
        return "bg-muted";
      case "error":
        return "bg-neon-red";
      default:
        return "bg-muted";
    }
  };

  const filteredLogs = logs.filter(
    (log) => filterLevel === "all" || log.level === filterLevel,
  );

  const handleStopExecution = () => {
    if (execution) {
      setExecution((prev) => (prev ? { ...prev, status: "cancelled" } : null));
    }
  };

  const handleDownloadReport = () => {
    // Generate and download execution report
    console.log("Downloading execution report...");
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

  if (!execution) {
    return (
      <div className="container mx-auto p-8">
        <div className="text-center py-12">
          <AlertCircle className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-2xl font-semibold mb-2">Execution Not Found</h2>
          <p className="text-muted-foreground mb-4">
            The execution with ID {executionId} could not be found.
          </p>
          <Button onClick={() => router.push("/playbooks")}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Playbooks
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
          <div>
            <h1 className="text-4xl font-bold text-gradient">
              {execution.playbook_name}
            </h1>
            <p className="text-xl text-muted-foreground">
              Execution Monitor - {execution.id.slice(0, 8)}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <Badge className={getStatusColor(execution.status)}>
            {execution.status.toUpperCase()}
          </Badge>

          {execution.status === "running" && (
            <Button
              variant="outline"
              onClick={handleStopExecution}
              className="text-neon-red border-neon-red hover:bg-neon-red/10"
            >
              <Square className="h-4 w-4 mr-2" />
              Stop
            </Button>
          )}

          {execution.status === "completed" && (
            <Button
              variant="outline"
              onClick={handleDownloadReport}
              className="btn-neon text-neon-lime"
            >
              <Download className="h-4 w-4 mr-2" />
              Report
            </Button>
          )}
        </div>
      </div>

      {/* Progress Overview */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5 text-neon-magenta" />
            Execution Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Overall Progress</span>
                <span className="text-sm text-muted-foreground">
                  {execution.progress_percentage.toFixed(1)}%
                </span>
              </div>
              <Progress value={execution.progress_percentage} className="h-3" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-muted-foreground">Started</div>
                <div className="font-medium">
                  {formatRelativeTime(execution.started_at)}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Duration</div>
                <div className="font-medium">
                  {execution.duration_seconds
                    ? formatDuration(execution.duration_seconds)
                    : formatDuration(
                        Math.floor(
                          (Date.now() -
                            new Date(execution.started_at).getTime()) /
                            1000,
                        ),
                      )}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">
                  Current Step
                </div>
                <div className="font-medium text-neon-lime">
                  {execution.current_step_name || "Initializing..."}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Steps</div>
                <div className="font-medium">
                  {
                    execution.steps.filter((s) => s.status === "completed")
                      .length
                  }{" "}
                  / {execution.steps.length}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Workflow Visualization */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Workflow Visualization</CardTitle>
              <CardDescription>
                Interactive workflow diagram with execution path tracking
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <WorkflowVisualization
                executionData={{
                  id: execution.id,
                  name: execution.playbook_name,
                  status: execution.status,
                  ...(execution.current_step && {
                    currentStep: execution.current_step,
                  }),
                  executionPath: execution.steps
                    .filter(
                      (step) =>
                        step.status === "completed" ||
                        step.status === "running",
                    )
                    .map((step) => step.id),
                  steps: execution.steps.map((step, index) => ({
                    id: step.id,
                    name: step.name,
                    type:
                      index === 0
                        ? ("start" as const)
                        : index === execution.steps.length - 1
                          ? ("end" as const)
                          : ("action" as const),
                    status: step.status,
                    progress: step.progress,
                    ...(step.duration !== undefined && {
                      duration: step.duration,
                    }),
                    ...(step.agent_name && {
                      agent_name: step.agent_name,
                    }),
                    position: {
                      x: 200 + (index % 3) * 250,
                      y: 100 + Math.floor(index / 3) * 150,
                    },
                    connections: (() => {
                      const nextStep = execution.steps[index + 1];
                      return index < execution.steps.length - 1 && nextStep
                        ? [nextStep.id]
                        : [];
                    })(),
                  })),
                  startTime: execution.started_at,
                  ...(execution.completed_at && {
                    endTime: execution.completed_at,
                  }),
                }}
                height={500}
                onStepSelect={(stepId) => {
                  // Handle step selection
                  console.log("Selected step:", stepId);
                }}
                className="rounded-lg"
              />
            </CardContent>
          </Card>

          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Step Progress</CardTitle>
              <CardDescription>
                Detailed step-by-step execution status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {execution.steps.map((step) => (
                  <div key={step.id} className="flex items-center space-x-4">
                    <div className="flex items-center justify-center w-8 h-8 rounded-full border-2 border-border">
                      {getStatusIcon(step.status)}
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium truncate">{step.name}</h4>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {step.agent_name}
                          </Badge>
                          {step.duration && (
                            <span className="text-xs text-muted-foreground">
                              {formatDuration(step.duration)}
                            </span>
                          )}
                        </div>
                      </div>

                      {step.status === "running" && (
                        <div className="mt-2">
                          <Progress value={step.progress} className="h-2" />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Live Logs */}
          <Card className="cyber-card">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Live Logs</CardTitle>
                  <CardDescription>
                    Real-time execution logs and system messages
                  </CardDescription>
                </div>
                <div className="flex items-center space-x-2">
                  <select
                    value={filterLevel}
                    onChange={(e) => setFilterLevel(e.target.value)}
                    className="text-sm bg-background border border-border rounded px-2 py-1"
                  >
                    <option value="all">All Levels</option>
                    <option value="info">Info</option>
                    <option value="warning">Warning</option>
                    <option value="error">Error</option>
                    <option value="debug">Debug</option>
                  </select>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setAutoScroll(!autoScroll)}
                    className={autoScroll ? "text-neon-lime" : ""}
                  >
                    Auto-scroll
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setLogsExpanded(!logsExpanded)}
                  >
                    {logsExpanded ? (
                      <Minimize2 className="h-4 w-4" />
                    ) : (
                      <Maximize2 className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div
                ref={logsContainerRef}
                className={cn(
                  "bg-cyber-dark-grey/30 rounded-lg p-4 font-mono text-sm overflow-y-auto",
                  logsExpanded ? "h-96" : "h-64",
                )}
              >
                {filteredLogs.map((log) => (
                  <div key={log.id} className="flex items-start space-x-3 mb-2">
                    <span className="text-muted-foreground text-xs shrink-0">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span
                      className={cn(
                        "text-xs font-medium shrink-0 w-16",
                        getLogLevelColor(log.level),
                      )}
                    >
                      [{log.level.toUpperCase()}]
                    </span>
                    {log.agent_name && (
                      <span className="text-neon-cyan text-xs shrink-0">
                        {log.agent_name}:
                      </span>
                    )}
                    <span className="text-foreground flex-1">
                      {log.message}
                    </span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Agents Status */}
        <div className="space-y-6">
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Agents Status</CardTitle>
              <CardDescription>
                Current status of all agents in this execution
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(execution.agents).map(([agentId, agent]) => (
                  <div
                    key={agentId}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-3">
                      <div
                        className={cn(
                          "w-3 h-3 rounded-full",
                          getAgentStatusColor(agent.status),
                        )}
                      ></div>
                      <span className="font-medium">{agent.name}</span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {agent.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Input Variables */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle>Input Variables</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(execution.input_variables).map(
                  ([key, value]) => (
                    <div key={key} className="text-sm">
                      <span className="text-muted-foreground">{key}:</span>
                      <span className="ml-2 font-mono text-neon-lime">
                        {typeof value === "string"
                          ? value
                          : JSON.stringify(value)}
                      </span>
                    </div>
                  ),
                )}
              </div>
            </CardContent>
          </Card>

          {execution.error_message && (
            <Card className="cyber-card border-neon-red">
              <CardHeader>
                <CardTitle className="text-neon-red flex items-center gap-2">
                  <AlertCircle className="h-5 w-5" />
                  Error
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-neon-red font-mono">
                  {execution.error_message}
                </p>
              </CardContent>
            </Card>
          )}

          {execution.status === "completed" && execution.output_data && (
            <Card className="cyber-card border-neon-green">
              <CardHeader>
                <CardTitle className="text-neon-green">Output Data</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-xs text-neon-green font-mono overflow-auto">
                  {JSON.stringify(execution.output_data, null, 2)}
                </pre>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
