"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Play,
  Pause,
  Square,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  BarChart3,
  Zap,
  Server,
  Eye,
  RefreshCw,
  Filter,
  Bell,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";
import MetricsDashboard from "@/components/monitor/MetricsDashboard";
import AlertSystem from "@/components/monitor/AlertSystem";
import SystemHealthDashboard from "@/components/monitor/SystemHealthDashboard";
import PerformanceCharts from "@/components/monitor/PerformanceCharts";
import ErrorTracking from "@/components/monitor/ErrorTracking";

// Types for execution monitoring
interface ExecutionMetrics {
  execution_time_seconds: number;
  progress_percentage: number;
  completed_steps: number;
  failed_steps: number;
  skipped_steps: number;
  total_steps: number;
  variables_count: number;
  checkpoints_count: number;
  average_step_duration: number;
  phase: string;
  is_paused: boolean;
  is_cancelled: boolean;
}

interface ExecutionContext {
  execution_id: string;
  workflow_id: string;
  phase: string;
  is_paused: boolean;
  is_cancelled: boolean;
  progress_percentage: number;
  variables: Record<string, any>;
  completed_steps: number[];
  failed_steps: number[];
  skipped_steps: number[];
  output_data: Record<string, any>;
  error_details: any;
  start_time: string;
  last_checkpoint: string | null;
  step_count: number;
  checkpoint_count: number;
}

interface PlaybookExecution {
  id: string;
  playbookId: string;
  playbookName: string;
  status:
    | "pending"
    | "running"
    | "completed"
    | "failed"
    | "paused"
    | "cancelled";
  startTime: string;
  endTime?: string;
  progress: number;
  currentStep?: string;
  totalSteps: number;
  completedSteps: number;
  failedSteps: number;
  metrics: ExecutionMetrics;
  context: ExecutionContext;
  logs: Array<{
    id: string;
    timestamp: string;
    level: "info" | "warn" | "error" | "debug";
    message: string;
    step?: string;
    data?: any;
  }>;
}

// Mock data for demonstration
const mockExecutions: PlaybookExecution[] = [
  {
    id: "exec-001",
    playbookId: "pb-001",
    playbookName: "User Onboarding Flow",
    status: "running",
    startTime: new Date(Date.now() - 300000).toISOString(),
    progress: 65,
    currentStep: "Send Welcome Email",
    totalSteps: 8,
    completedSteps: 5,
    failedSteps: 0,
    metrics: {
      execution_time_seconds: 300,
      progress_percentage: 65,
      completed_steps: 5,
      failed_steps: 0,
      skipped_steps: 0,
      total_steps: 8,
      variables_count: 12,
      checkpoints_count: 3,
      average_step_duration: 45.5,
      phase: "execution",
      is_paused: false,
      is_cancelled: false,
    },
    context: {
      execution_id: "exec-001",
      workflow_id: "pb-001",
      phase: "execution",
      is_paused: false,
      is_cancelled: false,
      progress_percentage: 65,
      variables: { userId: "user123", email: "user@example.com" },
      completed_steps: [1, 2, 3, 4, 5],
      failed_steps: [],
      skipped_steps: [],
      output_data: {},
      error_details: null,
      start_time: new Date(Date.now() - 300000).toISOString(),
      last_checkpoint: new Date(Date.now() - 60000).toISOString(),
      step_count: 5,
      checkpoint_count: 3,
    },
    logs: [
      {
        id: "log-001",
        timestamp: new Date(Date.now() - 60000).toISOString(),
        level: "info",
        message: "Step completed: Validate User Data",
        step: "validate-user",
      },
      {
        id: "log-002",
        timestamp: new Date(Date.now() - 30000).toISOString(),
        level: "info",
        message: "Starting: Send Welcome Email",
        step: "send-email",
      },
    ],
  },
  {
    id: "exec-002",
    playbookId: "pb-002",
    playbookName: "Data Processing Pipeline",
    status: "completed",
    startTime: new Date(Date.now() - 1800000).toISOString(),
    endTime: new Date(Date.now() - 300000).toISOString(),
    progress: 100,
    totalSteps: 12,
    completedSteps: 12,
    failedSteps: 0,
    metrics: {
      execution_time_seconds: 1500,
      progress_percentage: 100,
      completed_steps: 12,
      failed_steps: 0,
      skipped_steps: 0,
      total_steps: 12,
      variables_count: 25,
      checkpoints_count: 8,
      average_step_duration: 125.0,
      phase: "completion",
      is_paused: false,
      is_cancelled: false,
    },
    context: {
      execution_id: "exec-002",
      workflow_id: "pb-002",
      phase: "completion",
      is_paused: false,
      is_cancelled: false,
      progress_percentage: 100,
      variables: { batchId: "batch456", recordCount: 10000 },
      completed_steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      failed_steps: [],
      skipped_steps: [],
      output_data: { processedRecords: 10000, errors: 0 },
      error_details: null,
      start_time: new Date(Date.now() - 1800000).toISOString(),
      last_checkpoint: new Date(Date.now() - 300000).toISOString(),
      step_count: 12,
      checkpoint_count: 8,
    },
    logs: [
      {
        id: "log-003",
        timestamp: new Date(Date.now() - 300000).toISOString(),
        level: "info",
        message: "Pipeline completed successfully",
      },
    ],
  },
  {
    id: "exec-003",
    playbookId: "pb-003",
    playbookName: "Security Audit",
    status: "failed",
    startTime: new Date(Date.now() - 900000).toISOString(),
    endTime: new Date(Date.now() - 600000).toISOString(),
    progress: 40,
    totalSteps: 6,
    completedSteps: 2,
    failedSteps: 1,
    metrics: {
      execution_time_seconds: 300,
      progress_percentage: 40,
      completed_steps: 2,
      failed_steps: 1,
      skipped_steps: 0,
      total_steps: 6,
      variables_count: 8,
      checkpoints_count: 2,
      average_step_duration: 100.0,
      phase: "error_handling",
      is_paused: false,
      is_cancelled: false,
    },
    context: {
      execution_id: "exec-003",
      workflow_id: "pb-003",
      phase: "error_handling",
      is_paused: false,
      is_cancelled: false,
      progress_percentage: 40,
      variables: { targetSystem: "prod-api" },
      completed_steps: [1, 2],
      failed_steps: [3],
      skipped_steps: [],
      output_data: {},
      error_details: {
        error_type: "ConnectionError",
        error_message: "Failed to connect to target system",
        step_id: 3,
        timestamp: new Date(Date.now() - 600000).toISOString(),
        phase: "execution",
      },
      start_time: new Date(Date.now() - 900000).toISOString(),
      last_checkpoint: new Date(Date.now() - 720000).toISOString(),
      step_count: 2,
      checkpoint_count: 2,
    },
    logs: [
      {
        id: "log-004",
        timestamp: new Date(Date.now() - 600000).toISOString(),
        level: "error",
        message: "Failed to connect to target system",
        step: "security-scan",
        data: { error: "ConnectionError", timeout: 30000 },
      },
    ],
  },
];

export default function MonitorPage() {
  const router = useRouter();
  const [executions, setExecutions] =
    useState<PlaybookExecution[]>(mockExecutions);
  const [selectedExecution] = useState<PlaybookExecution | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<
    "executions" | "metrics" | "alerts" | "health" | "performance" | "errors"
  >("health");

  // Auto-refresh executions
  useEffect(() => {
    const interval = setInterval(() => {
      // In a real implementation, this would fetch from API
      setExecutions((prev) =>
        prev.map((exec) => {
          if (exec.status === "running") {
            // Simulate progress
            const newProgress = Math.min(
              exec.progress + Math.random() * 5,
              100,
            );
            const newCompletedSteps = Math.floor(
              (newProgress / 100) * exec.totalSteps,
            );

            return {
              ...exec,
              progress: newProgress,
              completedSteps: newCompletedSteps,
              metrics: {
                ...exec.metrics,
                progress_percentage: newProgress,
                completed_steps: newCompletedSteps,
                execution_time_seconds: exec.metrics.execution_time_seconds + 5,
              },
              context: {
                ...exec.context,
                progress_percentage: newProgress,
                completed_steps: Array.from(
                  { length: newCompletedSteps },
                  (_, i) => i + 1,
                ),
              },
            };
          }
          return exec;
        }),
      );
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsRefreshing(false);
  };

  const getStatusIcon = (status: PlaybookExecution["status"]) => {
    switch (status) {
      case "running":
        return <Play className="h-4 w-4 text-neon-green animate-pulse" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-neon-green" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      case "paused":
        return <Pause className="h-4 w-4 text-neon-yellow" />;
      case "cancelled":
        return <Square className="h-4 w-4 text-neon-orange" />;
      default:
        return <Clock className="h-4 w-4 text-neon-cyan" />;
    }
  };

  const getStatusColor = (status: PlaybookExecution["status"]) => {
    switch (status) {
      case "running":
        return "text-neon-green border-neon-green/50 bg-neon-green/10";
      case "completed":
        return "text-neon-green border-neon-green/50 bg-neon-green/10";
      case "failed":
        return "text-neon-red border-neon-red/50 bg-neon-red/10";
      case "paused":
        return "text-neon-yellow border-neon-yellow/50 bg-neon-yellow/10";
      case "cancelled":
        return "text-neon-orange border-neon-orange/50 bg-neon-orange/10";
      default:
        return "text-neon-cyan border-neon-cyan/50 bg-neon-cyan/10";
    }
  };

  const filteredExecutions = executions.filter((exec) => {
    if (filter === "all") return true;
    return exec.status === filter;
  });

  const runningCount = executions.filter((e) => e.status === "running").length;
  const completedCount = executions.filter(
    (e) => e.status === "completed",
  ).length;
  const failedCount = executions.filter((e) => e.status === "failed").length;

  return (
    <div className="min-h-screen bg-cyber-dark text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-neon-cyan">
              Execution Monitor
            </h1>
            <p className="text-neon-cyan/70 mt-1">
              Real-time monitoring and analytics dashboard
            </p>
          </div>
          <div className="flex items-center gap-4">
            <Button
              onClick={handleRefresh}
              disabled={isRefreshing}
              variant="outline"
              size="sm"
              className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
            >
              <RefreshCw
                className={cn("h-4 w-4 mr-2", isRefreshing && "animate-spin")}
              />
              Refresh
            </Button>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="bg-cyber-dark border-neon-green/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <Activity className="h-8 w-8 text-neon-green" />
                <div>
                  <p className="text-sm text-neon-green/70">Running</p>
                  <p className="text-2xl font-bold text-neon-green">
                    {runningCount}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-cyber-dark border-neon-green/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <CheckCircle className="h-8 w-8 text-neon-green" />
                <div>
                  <p className="text-sm text-neon-green/70">Completed</p>
                  <p className="text-2xl font-bold text-neon-green">
                    {completedCount}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-cyber-dark border-neon-red/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <XCircle className="h-8 w-8 text-neon-red" />
                <div>
                  <p className="text-sm text-neon-red/70">Failed</p>
                  <p className="text-2xl font-bold text-neon-red">
                    {failedCount}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-cyber-dark border-neon-cyan/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <BarChart3 className="h-8 w-8 text-neon-cyan" />
                <div>
                  <p className="text-sm text-neon-cyan/70">Total</p>
                  <p className="text-2xl font-bold text-neon-cyan">
                    {executions.length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-1 border-b border-neon-cyan/20">
          {[
            { id: "health", label: "System Health", icon: Server },
            { id: "performance", label: "Performance", icon: BarChart3 },
            { id: "errors", label: "Error Tracking", icon: AlertTriangle },
            { id: "executions", label: "Executions", icon: Activity },
            { id: "metrics", label: "Metrics", icon: TrendingUp },
            { id: "alerts", label: "Alerts", icon: Bell },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                "flex items-center gap-2 px-6 py-3 text-sm font-medium transition-colors",
                activeTab === tab.id
                  ? "text-neon-cyan border-b-2 border-neon-cyan bg-neon-cyan/5"
                  : "text-neon-cyan/70 hover:text-neon-cyan hover:bg-neon-cyan/5",
              )}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === "health" && <SystemHealthDashboard />}

        {activeTab === "performance" && <PerformanceCharts />}

        {activeTab === "errors" && <ErrorTracking />}

        {activeTab === "executions" && (
          <div className="space-y-6">
            {/* Filter Bar */}
            <div className="flex items-center gap-4">
              <Filter className="h-5 w-5 text-neon-cyan" />
              <div className="flex gap-2">
                {["all", "running", "completed", "failed", "paused"].map(
                  (status) => (
                    <Button
                      key={status}
                      onClick={() => setFilter(status)}
                      variant={filter === status ? "default" : "outline"}
                      size="sm"
                      className={cn(
                        "capitalize",
                        filter === status
                          ? "bg-neon-cyan text-cyber-dark"
                          : "border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10",
                      )}
                    >
                      {status}
                    </Button>
                  ),
                )}
              </div>
            </div>

            {/* Executions Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {filteredExecutions.map((execution) => (
                <Card
                  key={execution.id}
                  className={cn(
                    "bg-cyber-dark border transition-all duration-200 cursor-pointer hover:shadow-lg",
                    selectedExecution?.id === execution.id
                      ? "border-neon-cyan shadow-neon-cyan/50"
                      : "border-neon-cyan/20 hover:border-neon-cyan/40",
                  )}
                  onClick={() => router.push(`/monitor/${execution.id}`)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg text-white">
                        {execution.playbookName}
                      </CardTitle>
                      <Badge
                        variant="outline"
                        className={cn(
                          "text-xs",
                          getStatusColor(execution.status),
                        )}
                      >
                        <div className="flex items-center gap-1">
                          {getStatusIcon(execution.status)}
                          {execution.status.toUpperCase()}
                        </div>
                      </Badge>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* Progress */}
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-neon-cyan/70">Progress</span>
                        <span className="text-neon-cyan">
                          {execution.progress.toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={execution.progress}
                        className="h-2 bg-cyber-dark"
                      />
                    </div>

                    {/* Steps Info */}
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="text-neon-green/70">Completed</p>
                        <p className="text-neon-green font-semibold">
                          {execution.completedSteps}
                        </p>
                      </div>
                      <div>
                        <p className="text-neon-red/70">Failed</p>
                        <p className="text-neon-red font-semibold">
                          {execution.failedSteps}
                        </p>
                      </div>
                      <div>
                        <p className="text-neon-cyan/70">Total</p>
                        <p className="text-neon-cyan font-semibold">
                          {execution.totalSteps}
                        </p>
                      </div>
                    </div>

                    {/* Current Step */}
                    {execution.currentStep && (
                      <div className="flex items-center gap-2 text-sm">
                        <Zap className="h-4 w-4 text-neon-yellow" />
                        <span className="text-neon-yellow">
                          {execution.currentStep}
                        </span>
                      </div>
                    )}

                    {/* Timing */}
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-neon-cyan/70" />
                        <span className="text-neon-cyan/70">
                          {new Date(execution.startTime).toLocaleTimeString()}
                        </span>
                      </div>
                      {execution.endTime && (
                        <span className="text-neon-cyan/70">
                          Duration:{" "}
                          {Math.round(
                            (new Date(execution.endTime).getTime() -
                              new Date(execution.startTime).getTime()) /
                              1000,
                          )}
                          s
                        </span>
                      )}
                    </div>

                    {/* Error Details */}
                    {execution.context.error_details && (
                      <div className="flex items-start gap-2 p-3 bg-neon-red/10 border border-neon-red/20 rounded">
                        <AlertTriangle className="h-4 w-4 text-neon-red mt-0.5 flex-shrink-0" />
                        <div className="text-sm">
                          <p className="text-neon-red font-semibold">
                            {execution.context.error_details.error_type}
                          </p>
                          <p className="text-neon-red/80">
                            {execution.context.error_details.error_message}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-2 pt-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation();
                          router.push(`/monitor/${execution.id}`);
                        }}
                        className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        Details
                      </Button>
                      {execution.status === "running" && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-neon-yellow/50 text-neon-yellow hover:bg-neon-yellow/10"
                        >
                          <Pause className="h-4 w-4 mr-1" />
                          Pause
                        </Button>
                      )}
                      {execution.status === "paused" && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-neon-green/50 text-neon-green hover:bg-neon-green/10"
                        >
                          <Play className="h-4 w-4 mr-1" />
                          Resume
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {filteredExecutions.length === 0 && (
              <Card className="bg-cyber-dark border-neon-cyan/20">
                <CardContent className="p-12 text-center">
                  <Server className="h-12 w-12 text-neon-cyan/50 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-neon-cyan mb-2">
                    No executions found
                  </h3>
                  <p className="text-neon-cyan/70">
                    {filter === "all"
                      ? "No playbook executions are currently active."
                      : `No ${filter} executions found.`}
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {activeTab === "metrics" && <MetricsDashboard refreshInterval={5000} />}

        {activeTab === "alerts" && (
          <AlertSystem
            maxAlerts={100}
            autoRefresh={true}
            refreshInterval={5000}
          />
        )}
      </div>
    </div>
  );
}
