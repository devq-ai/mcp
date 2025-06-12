"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

import {
  Activity,
  TrendingUp,
  TrendingDown,
  Clock,
  Zap,
  Server,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Cpu,
  HardDrive,
  Network,
  PlayCircle,
  PauseCircle,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricValue {
  current: number;
  previous: number;
  unit: string;
  trend: "up" | "down" | "stable";
}

interface SystemMetrics {
  cpu: MetricValue;
  memory: MetricValue;
  disk: MetricValue;
  network: MetricValue;
}

interface ExecutionMetrics {
  activeExecutions: number;
  completedExecutions: number;
  failedExecutions: number;
  pausedExecutions: number;
  averageExecutionTime: MetricValue;
  throughput: MetricValue;
  errorRate: MetricValue;
  successRate: MetricValue;
}

interface PerformanceData {
  timestamp: string;
  value: number;
}

interface MetricsDashboardProps {
  refreshInterval?: number;
  className?: string;
}

export default function MetricsDashboard({
  refreshInterval = 5000,
  className,
}: MetricsDashboardProps) {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu: { current: 45.2, previous: 42.8, unit: "%", trend: "up" },
    memory: { current: 68.5, previous: 65.1, unit: "%", trend: "up" },
    disk: { current: 23.7, previous: 23.5, unit: "%", trend: "up" },
    network: { current: 156.8, previous: 142.3, unit: "MB/s", trend: "up" },
  });

  const [executionMetrics, setExecutionMetrics] = useState<ExecutionMetrics>({
    activeExecutions: 12,
    completedExecutions: 348,
    failedExecutions: 8,
    pausedExecutions: 2,
    averageExecutionTime: {
      current: 45.6,
      previous: 48.2,
      unit: "s",
      trend: "down",
    },
    throughput: {
      current: 23.4,
      previous: 21.8,
      unit: "exec/min",
      trend: "up",
    },
    errorRate: { current: 2.1, previous: 2.8, unit: "%", trend: "down" },
    successRate: { current: 97.9, previous: 97.2, unit: "%", trend: "up" },
  });

  const [performanceHistory, setPerformanceHistory] = useState<
    PerformanceData[]
  >([
    { timestamp: "10:00", value: 85 },
    { timestamp: "10:05", value: 89 },
    { timestamp: "10:10", value: 76 },
    { timestamp: "10:15", value: 92 },
    { timestamp: "10:20", value: 88 },
    { timestamp: "10:25", value: 94 },
    { timestamp: "10:30", value: 91 },
  ]);

  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [alerts] = useState([
    {
      id: "alert-1",
      type: "warning" as const,
      message: "High memory usage detected (68.5%)",
      timestamp: new Date(Date.now() - 120000),
      metric: "memory",
    },
    {
      id: "alert-2",
      type: "info" as const,
      message: "Execution throughput increased by 7.3%",
      timestamp: new Date(Date.now() - 300000),
      metric: "throughput",
    },
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update system metrics
      setSystemMetrics((prev) => ({
        cpu: {
          ...prev.cpu,
          previous: prev.cpu.current,
          current: Math.max(
            0,
            Math.min(100, prev.cpu.current + (Math.random() - 0.5) * 5),
          ),
          trend: Math.random() > 0.5 ? "up" : "down",
        },
        memory: {
          ...prev.memory,
          previous: prev.memory.current,
          current: Math.max(
            0,
            Math.min(100, prev.memory.current + (Math.random() - 0.5) * 3),
          ),
          trend: Math.random() > 0.5 ? "up" : "down",
        },
        disk: {
          ...prev.disk,
          previous: prev.disk.current,
          current: Math.max(
            0,
            Math.min(100, prev.disk.current + (Math.random() - 0.5) * 1),
          ),
          trend: Math.random() > 0.5 ? "up" : "down",
        },
        network: {
          ...prev.network,
          previous: prev.network.current,
          current: Math.max(
            0,
            prev.network.current + (Math.random() - 0.5) * 20,
          ),
          trend: Math.random() > 0.5 ? "up" : "down",
        },
      }));

      // Update execution metrics
      setExecutionMetrics((prev) => ({
        ...prev,
        activeExecutions: Math.max(
          0,
          prev.activeExecutions + Math.floor((Math.random() - 0.5) * 3),
        ),
        averageExecutionTime: {
          ...prev.averageExecutionTime,
          previous: prev.averageExecutionTime.current,
          current: Math.max(
            10,
            prev.averageExecutionTime.current + (Math.random() - 0.5) * 5,
          ),
          trend: Math.random() > 0.5 ? "up" : "down",
        },
        throughput: {
          ...prev.throughput,
          previous: prev.throughput.current,
          current: Math.max(
            0,
            prev.throughput.current + (Math.random() - 0.5) * 2,
          ),
          trend: Math.random() > 0.5 ? "up" : "down",
        },
        errorRate: {
          ...prev.errorRate,
          previous: prev.errorRate.current,
          current: Math.max(
            0,
            Math.min(10, prev.errorRate.current + (Math.random() - 0.5) * 0.5),
          ),
          trend: Math.random() > 0.5 ? "up" : "down",
        },
        successRate: {
          ...prev.successRate,
          previous: prev.successRate.current,
          current: Math.max(90, Math.min(100, 100 - prev.errorRate.current)),
          trend: prev.errorRate.trend === "up" ? "down" : "up",
        },
      }));

      // Update performance history
      setPerformanceHistory((prev) => {
        const newValue = Math.max(
          0,
          Math.min(100, 85 + (Math.random() - 0.5) * 20),
        );
        const newEntry = {
          timestamp: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          value: newValue,
        };
        return [...prev.slice(-6), newEntry];
      });

      setLastUpdated(new Date());
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return <TrendingUp className="h-4 w-4 text-neon-red" />;
      case "down":
        return <TrendingDown className="h-4 w-4 text-neon-green" />;
      default:
        return <Activity className="h-4 w-4 text-neon-cyan" />;
    }
  };

  const getTrendColor = (
    trend: "up" | "down" | "stable",
    isGoodWhenUp: boolean = true,
  ) => {
    if (trend === "stable") return "text-neon-cyan";
    const isGood = isGoodWhenUp ? trend === "up" : trend === "down";
    return isGood ? "text-neon-green" : "text-neon-red";
  };

  const getAlertIcon = (type: "error" | "warning" | "info") => {
    switch (type) {
      case "error":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-neon-yellow" />;
      default:
        return <CheckCircle className="h-4 w-4 text-neon-cyan" />;
    }
  };

  const getHealthStatus = () => {
    const criticalAlerts = 0; // No critical alerts in mock data
    const warnings = alerts.filter((a) => a.type === "warning").length;

    if (criticalAlerts > 0) return { status: "critical", color: "neon-red" };
    if (warnings > 0) return { status: "warning", color: "neon-yellow" };
    return { status: "healthy", color: "neon-green" };
  };

  const health = getHealthStatus();

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-neon-cyan">System Metrics</h2>
          <p className="text-neon-cyan/70">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge
            variant="outline"
            className={cn(
              "text-sm",
              `text-${health.color} border-${health.color}/50 bg-${health.color}/10`,
            )}
          >
            <div className="flex items-center gap-2">
              <div
                className={cn("w-2 h-2 rounded-full", `bg-${health.color}`)}
              />
              {health.status.toUpperCase()}
            </div>
          </Badge>
        </div>
      </div>

      {/* System Resources */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-cyber-dark border-neon-cyan/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-cyan flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                CPU Usage
              </CardTitle>
              {getTrendIcon(systemMetrics.cpu.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-white">
                {systemMetrics.cpu.current.toFixed(1)}
                {systemMetrics.cpu.unit}
              </div>
              <Progress value={systemMetrics.cpu.current} className="h-2" />
              <div
                className={cn(
                  "text-sm flex items-center gap-1",
                  getTrendColor(systemMetrics.cpu.trend, false),
                )}
              >
                {systemMetrics.cpu.trend === "up" ? "+" : ""}
                {(
                  systemMetrics.cpu.current - systemMetrics.cpu.previous
                ).toFixed(1)}
                {systemMetrics.cpu.unit} from previous
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-cyan flex items-center gap-2">
                <Server className="h-4 w-4" />
                Memory
              </CardTitle>
              {getTrendIcon(systemMetrics.memory.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-white">
                {systemMetrics.memory.current.toFixed(1)}
                {systemMetrics.memory.unit}
              </div>
              <Progress value={systemMetrics.memory.current} className="h-2" />
              <div
                className={cn(
                  "text-sm flex items-center gap-1",
                  getTrendColor(systemMetrics.memory.trend, false),
                )}
              >
                {systemMetrics.memory.trend === "up" ? "+" : ""}
                {(
                  systemMetrics.memory.current - systemMetrics.memory.previous
                ).toFixed(1)}
                {systemMetrics.memory.unit} from previous
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-cyan flex items-center gap-2">
                <HardDrive className="h-4 w-4" />
                Disk Usage
              </CardTitle>
              {getTrendIcon(systemMetrics.disk.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-white">
                {systemMetrics.disk.current.toFixed(1)}
                {systemMetrics.disk.unit}
              </div>
              <Progress value={systemMetrics.disk.current} className="h-2" />
              <div
                className={cn(
                  "text-sm flex items-center gap-1",
                  getTrendColor(systemMetrics.disk.trend, false),
                )}
              >
                {systemMetrics.disk.trend === "up" ? "+" : ""}
                {(
                  systemMetrics.disk.current - systemMetrics.disk.previous
                ).toFixed(1)}
                {systemMetrics.disk.unit} from previous
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-cyan flex items-center gap-2">
                <Network className="h-4 w-4" />
                Network I/O
              </CardTitle>
              {getTrendIcon(systemMetrics.network.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-white">
                {systemMetrics.network.current.toFixed(1)}
                {systemMetrics.network.unit}
              </div>
              <div className="h-2 bg-neon-cyan/20 rounded-full overflow-hidden">
                <div
                  className="h-full bg-neon-cyan transition-all duration-1000"
                  style={{
                    width: `${Math.min(100, (systemMetrics.network.current / 200) * 100)}%`,
                  }}
                />
              </div>
              <div
                className={cn(
                  "text-sm flex items-center gap-1",
                  getTrendColor(systemMetrics.network.trend, true),
                )}
              >
                {systemMetrics.network.trend === "up" ? "+" : ""}
                {(
                  systemMetrics.network.current - systemMetrics.network.previous
                ).toFixed(1)}
                {systemMetrics.network.unit} from previous
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Execution Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-cyber-dark border-neon-green/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-neon-green flex items-center gap-2">
              <PlayCircle className="h-4 w-4" />
              Active Executions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-green">
              {executionMetrics.activeExecutions}
            </div>
            <p className="text-xs text-neon-green/70">Currently running</p>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-green/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-neon-green flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              Completed Today
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-green">
              {executionMetrics.completedExecutions}
            </div>
            <p className="text-xs text-neon-green/70">Successfully completed</p>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-red/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-neon-red flex items-center gap-2">
              <XCircle className="h-4 w-4" />
              Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-red">
              {executionMetrics.failedExecutions}
            </div>
            <p className="text-xs text-neon-red/70">With errors</p>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-yellow/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-neon-yellow flex items-center gap-2">
              <PauseCircle className="h-4 w-4" />
              Paused
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-yellow">
              {executionMetrics.pausedExecutions}
            </div>
            <p className="text-xs text-neon-yellow/70">Temporarily paused</p>
          </CardContent>
        </Card>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-cyber-dark border-neon-cyan/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-cyan flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Avg Execution Time
              </CardTitle>
              {getTrendIcon(executionMetrics.averageExecutionTime.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {executionMetrics.averageExecutionTime.current.toFixed(1)}
              {executionMetrics.averageExecutionTime.unit}
            </div>
            <div
              className={cn(
                "text-sm flex items-center gap-1",
                getTrendColor(
                  executionMetrics.averageExecutionTime.trend,
                  false,
                ),
              )}
            >
              {executionMetrics.averageExecutionTime.trend === "up" ? "+" : ""}
              {(
                executionMetrics.averageExecutionTime.current -
                executionMetrics.averageExecutionTime.previous
              ).toFixed(1)}
              {executionMetrics.averageExecutionTime.unit} from previous
            </div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-cyan flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Throughput
              </CardTitle>
              {getTrendIcon(executionMetrics.throughput.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {executionMetrics.throughput.current.toFixed(1)}
              {executionMetrics.throughput.unit}
            </div>
            <div
              className={cn(
                "text-sm flex items-center gap-1",
                getTrendColor(executionMetrics.throughput.trend, true),
              )}
            >
              {executionMetrics.throughput.trend === "up" ? "+" : ""}
              {(
                executionMetrics.throughput.current -
                executionMetrics.throughput.previous
              ).toFixed(1)}
              {executionMetrics.throughput.unit} from previous
            </div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-red/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-red flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Error Rate
              </CardTitle>
              {getTrendIcon(executionMetrics.errorRate.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {executionMetrics.errorRate.current.toFixed(1)}
              {executionMetrics.errorRate.unit}
            </div>
            <div
              className={cn(
                "text-sm flex items-center gap-1",
                getTrendColor(executionMetrics.errorRate.trend, false),
              )}
            >
              {executionMetrics.errorRate.trend === "up" ? "+" : ""}
              {(
                executionMetrics.errorRate.current -
                executionMetrics.errorRate.previous
              ).toFixed(1)}
              {executionMetrics.errorRate.unit} from previous
            </div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-green/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-neon-green flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                Success Rate
              </CardTitle>
              {getTrendIcon(executionMetrics.successRate.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {executionMetrics.successRate.current.toFixed(1)}
              {executionMetrics.successRate.unit}
            </div>
            <div
              className={cn(
                "text-sm flex items-center gap-1",
                getTrendColor(executionMetrics.successRate.trend, true),
              )}
            >
              {executionMetrics.successRate.trend === "up" ? "+" : ""}
              {(
                executionMetrics.successRate.current -
                executionMetrics.successRate.previous
              ).toFixed(1)}
              {executionMetrics.successRate.unit} from previous
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Chart */}
      <Card className="bg-cyber-dark border-neon-cyan/20">
        <CardHeader>
          <CardTitle className="text-neon-cyan flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Performance Trend
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-end justify-between gap-2">
            {performanceHistory.map((data, index) => (
              <div
                key={index}
                className="flex flex-col items-center gap-2 flex-1"
              >
                <div
                  className="w-full bg-neon-cyan rounded-t transition-all duration-1000 min-h-[4px]"
                  style={{ height: `${(data.value / 100) * 200}px` }}
                />
                <div className="text-xs text-neon-cyan/70 transform -rotate-45 origin-center">
                  {data.timestamp}
                </div>
                <div className="text-xs text-neon-cyan font-semibold">
                  {data.value.toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Alerts */}
      {alerts.length > 0 && (
        <Card className="bg-cyber-dark border-neon-yellow/20">
          <CardHeader>
            <CardTitle className="text-neon-yellow flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              Recent Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-start gap-3 p-3 bg-cyber-dark/50 border border-neon-yellow/10 rounded"
                >
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <p className="text-sm text-white">{alert.message}</p>
                    <p className="text-xs text-neon-cyan/50 mt-1">
                      {alert.timestamp.toLocaleTimeString()} • {alert.metric}
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-xs",
                      alert.type === "warning"
                        ? "text-neon-yellow border-neon-yellow/50"
                        : "text-neon-cyan border-neon-cyan/50",
                    )}
                  >
                    {alert.type.toUpperCase()}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
