"use client";

import * as React from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Cpu,
  MemoryStick,
  Network,
  Clock,
  CheckCircle,
  AlertTriangle,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricsData {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  response_time: number;
  success_rate: number;
  active_tasks: number;
  errors: number;
}

interface AgentMetrics {
  agent_id: string;
  agent_name: string;
  current_metrics: {
    cpu_usage: number;
    memory_usage: number;
    network_io: number;
    disk_usage: number;
    success_rate: number;
    avg_response_time: number;
    active_tasks: number;
    queue_length: number;
    errors_24h: number;
    uptime: number;
  };
  historical_data: MetricsData[];
  health_score: number;
  status: "active" | "idle" | "busy" | "error" | "offline";
}

interface AgentMetricsDashboardProps {
  metrics: AgentMetrics;
  timeRange?: "1h" | "6h" | "24h" | "7d";
  onTimeRangeChange?: (range: "1h" | "6h" | "24h" | "7d") => void;
}

export function AgentMetricsDashboard({
  metrics,
  timeRange = "24h",
  onTimeRangeChange,
}: AgentMetricsDashboardProps) {
  const [selectedMetric, setSelectedMetric] = React.useState<
    "cpu" | "memory" | "response" | "success"
  >("cpu");

  // Colors for charts
  const colors = {
    primary: "#FF0090",
    secondary: "#C7EA46",
    accent: "#FF5F1F",
    success: "#39FF14",
    warning: "#E9FF32",
    error: "#FF3131",
    info: "#00FFFF",
  };

  // Pie chart data for resource usage
  const resourceData = [
    {
      name: "CPU",
      value: metrics.current_metrics.cpu_usage,
      color: colors.primary,
    },
    {
      name: "Memory",
      value: metrics.current_metrics.memory_usage,
      color: colors.secondary,
    },
    {
      name: "Network",
      value: metrics.current_metrics.network_io * 10,
      color: colors.accent,
    },
    {
      name: "Disk",
      value: metrics.current_metrics.disk_usage,
      color: colors.info,
    },
  ];

  // Status distribution for task execution
  const taskStatusData = [
    {
      name: "Active",
      value: metrics.current_metrics.active_tasks,
      color: colors.success,
    },
    {
      name: "Queued",
      value: metrics.current_metrics.queue_length,
      color: colors.warning,
    },
    {
      name: "Errors",
      value: metrics.current_metrics.errors_24h,
      color: colors.error,
    },
  ];

  const getHealthColor = (score: number) => {
    if (score >= 90) return "text-neon-green";
    if (score >= 75) return "text-neon-lime";
    if (score >= 60) return "text-neon-orange";
    return "text-neon-red";
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active":
        return <CheckCircle className="h-5 w-5 text-neon-green" />;
      case "busy":
        return <Activity className="h-5 w-5 text-neon-lime animate-pulse" />;
      case "idle":
        return <Clock className="h-5 w-5 text-muted-foreground" />;
      case "error":
        return <XCircle className="h-5 w-5 text-neon-red" />;
      case "offline":
        return <XCircle className="h-5 w-5 text-muted" />;
      default:
        return <Clock className="h-5 w-5 text-muted-foreground" />;
    }
  };

  const calculateTrend = (data: MetricsData[], metric: keyof MetricsData) => {
    if (data.length < 2) return { direction: "stable", percentage: 0 };

    const recent = data.slice(-6); // Last 6 data points
    const older = data.slice(-12, -6); // Previous 6 data points

    const recentAvg =
      recent.reduce((sum, item) => sum + (item[metric] as number), 0) /
      recent.length;
    const olderAvg =
      older.reduce((sum, item) => sum + (item[metric] as number), 0) /
      older.length;

    const change = ((recentAvg - olderAvg) / olderAvg) * 100;

    return {
      direction: change > 2 ? "up" : change < -2 ? "down" : "stable",
      percentage: Math.abs(change),
    };
  };

  const cpuTrend = calculateTrend(metrics.historical_data, "cpu_usage");
  const memoryTrend = calculateTrend(metrics.historical_data, "memory_usage");
  const responseTrend = calculateTrend(
    metrics.historical_data,
    "response_time",
  );
  const successTrend = calculateTrend(metrics.historical_data, "success_rate");

  const formatTooltipValue = (value: number, name: string) => {
    switch (name) {
      case "CPU Usage":
      case "Memory Usage":
      case "Success Rate":
        return [`${value.toFixed(1)}%`, name];
      case "Response Time":
        return [`${value.toFixed(0)}ms`, name];
      case "Active Tasks":
      case "Errors":
        return [value.toString(), name];
      default:
        return [value.toString(), name];
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium">{`Time: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {`${entry.name}: ${formatTooltipValue(entry.value, entry.name)[0]}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Header with Agent Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {getStatusIcon(metrics.status)}
          <div>
            <h2 className="text-2xl font-bold text-gradient">
              {metrics.agent_name}
            </h2>
            <p className="text-muted-foreground">
              Performance Metrics Dashboard
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="text-right">
            <div className="text-sm text-muted-foreground">Health Score</div>
            <div
              className={cn(
                "text-2xl font-bold",
                getHealthColor(metrics.health_score),
              )}
            >
              {metrics.health_score}%
            </div>
          </div>

          {/* Time Range Selector */}
          <div className="flex space-x-1 bg-muted/20 rounded-lg p-1">
            {["1h", "6h", "24h", "7d"].map((range) => (
              <Button
                key={range}
                size="sm"
                variant={timeRange === range ? "default" : "ghost"}
                onClick={() => onTimeRangeChange?.(range as any)}
                className={cn(
                  "text-xs",
                  timeRange === range && "bg-primary text-primary-foreground",
                )}
              >
                {range}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Cpu className="h-4 w-4 text-neon-magenta" />
                CPU Usage
              </CardTitle>
              {cpuTrend.direction !== "stable" && (
                <div className="flex items-center text-xs">
                  {cpuTrend.direction === "up" ? (
                    <TrendingUp className="h-3 w-3 text-neon-red mr-1" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-neon-green mr-1" />
                  )}
                  {cpuTrend.percentage.toFixed(1)}%
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-magenta">
              {metrics.current_metrics.cpu_usage.toFixed(1)}%
            </div>
            <Progress
              value={metrics.current_metrics.cpu_usage}
              className="mt-2 h-2"
            />
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <MemoryStick className="h-4 w-4 text-neon-lime" />
                Memory Usage
              </CardTitle>
              {memoryTrend.direction !== "stable" && (
                <div className="flex items-center text-xs">
                  {memoryTrend.direction === "up" ? (
                    <TrendingUp className="h-3 w-3 text-neon-red mr-1" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-neon-green mr-1" />
                  )}
                  {memoryTrend.percentage.toFixed(1)}%
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-lime">
              {metrics.current_metrics.memory_usage.toFixed(1)}%
            </div>
            <Progress
              value={metrics.current_metrics.memory_usage}
              className="mt-2 h-2"
            />
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Clock className="h-4 w-4 text-neon-orange" />
                Response Time
              </CardTitle>
              {responseTrend.direction !== "stable" && (
                <div className="flex items-center text-xs">
                  {responseTrend.direction === "up" ? (
                    <TrendingUp className="h-3 w-3 text-neon-red mr-1" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-neon-green mr-1" />
                  )}
                  {responseTrend.percentage.toFixed(1)}%
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-orange">
              {metrics.current_metrics.avg_response_time}ms
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Average response time
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-neon-green" />
                Success Rate
              </CardTitle>
              {successTrend.direction !== "stable" && (
                <div className="flex items-center text-xs">
                  {successTrend.direction === "up" ? (
                    <TrendingUp className="h-3 w-3 text-neon-green mr-1" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-neon-red mr-1" />
                  )}
                  {successTrend.percentage.toFixed(1)}%
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-green">
              {metrics.current_metrics.success_rate.toFixed(1)}%
            </div>
            <Progress
              value={metrics.current_metrics.success_rate}
              className="mt-2 h-2"
            />
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Trends */}
        <Card className="cyber-card lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Performance Trends</CardTitle>
                <CardDescription>
                  CPU, Memory, and Response Time over {timeRange}
                </CardDescription>
              </div>
              <div className="flex space-x-2">
                {[
                  { key: "cpu", label: "CPU", icon: Cpu },
                  { key: "memory", label: "Memory", icon: MemoryStick },
                  { key: "response", label: "Response", icon: Clock },
                  { key: "success", label: "Success", icon: CheckCircle },
                ].map(({ key, label, icon: Icon }) => (
                  <Button
                    key={key}
                    size="sm"
                    variant={selectedMetric === key ? "default" : "outline"}
                    onClick={() => setSelectedMetric(key as any)}
                    className="text-xs"
                  >
                    <Icon className="h-3 w-3 mr-1" />
                    {label}
                  </Button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics.historical_data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2C2F33" />
                  <XAxis
                    dataKey="timestamp"
                    stroke="#7D8B99"
                    fontSize={12}
                    tickFormatter={(value) =>
                      new Date(value).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    }
                  />
                  <YAxis stroke="#7D8B99" fontSize={12} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />

                  {selectedMetric === "cpu" && (
                    <Line
                      type="monotone"
                      dataKey="cpu_usage"
                      stroke={colors.primary}
                      strokeWidth={2}
                      dot={{ fill: colors.primary, r: 3 }}
                      name="CPU Usage"
                    />
                  )}

                  {selectedMetric === "memory" && (
                    <Line
                      type="monotone"
                      dataKey="memory_usage"
                      stroke={colors.secondary}
                      strokeWidth={2}
                      dot={{ fill: colors.secondary, r: 3 }}
                      name="Memory Usage"
                    />
                  )}

                  {selectedMetric === "response" && (
                    <Line
                      type="monotone"
                      dataKey="response_time"
                      stroke={colors.accent}
                      strokeWidth={2}
                      dot={{ fill: colors.accent, r: 3 }}
                      name="Response Time"
                    />
                  )}

                  {selectedMetric === "success" && (
                    <Line
                      type="monotone"
                      dataKey="success_rate"
                      stroke={colors.success}
                      strokeWidth={2}
                      dot={{ fill: colors.success, r: 3 }}
                      name="Success Rate"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Resource Usage Distribution */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle>Resource Usage</CardTitle>
            <CardDescription>
              Current system resource distribution
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={resourceData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {resourceData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number) => [
                      `${value.toFixed(1)}%`,
                      "Usage",
                    ]}
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Task Status */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle>Task Status</CardTitle>
            <CardDescription>
              Current task execution distribution
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={taskStatusData} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" stroke="#2C2F33" />
                  <XAxis type="number" stroke="#7D8B99" fontSize={12} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    stroke="#7D8B99"
                    fontSize={12}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => [
                      value.toString(),
                      name,
                    ]}
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {taskStatusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Activity Summary */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle>Activity Summary</CardTitle>
          <CardDescription>
            Current agent activity and task information
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="flex items-center space-x-3">
              <Activity className="h-8 w-8 text-neon-lime" />
              <div>
                <div className="text-2xl font-bold">
                  {metrics.current_metrics.active_tasks}
                </div>
                <div className="text-sm text-muted-foreground">
                  Active Tasks
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <Clock className="h-8 w-8 text-neon-orange" />
              <div>
                <div className="text-2xl font-bold">
                  {metrics.current_metrics.queue_length}
                </div>
                <div className="text-sm text-muted-foreground">
                  Queued Tasks
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <Network className="h-8 w-8 text-neon-cyan" />
              <div>
                <div className="text-2xl font-bold">
                  {metrics.current_metrics.network_io.toFixed(1)}MB/s
                </div>
                <div className="text-sm text-muted-foreground">Network I/O</div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <AlertTriangle className="h-8 w-8 text-neon-red" />
              <div>
                <div className="text-2xl font-bold">
                  {metrics.current_metrics.errors_24h}
                </div>
                <div className="text-sm text-muted-foreground">
                  Errors (24h)
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
