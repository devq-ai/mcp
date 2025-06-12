"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart as PieChartIcon,
  Activity,
  Zap,
  Server,
  Download,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface PerformanceDataPoint {
  timestamp: string;
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  requests: number;
  responseTime: number;
  errors: number;
  activeUsers: number;
}

interface MetricSummary {
  metric: string;
  current: number;
  average: number;
  peak: number;
  unit: string;
  trend: "up" | "down" | "stable";
  change: number;
}

interface ExecutionStats {
  name: string;
  value: number;
  color: string;
}

interface PerformanceChartsProps {
  timeRange?: "1h" | "6h" | "24h" | "7d" | "30d";
  refreshInterval?: number;
  className?: string;
}

export function PerformanceCharts({
  timeRange = "6h",
  refreshInterval = 30000,
  className,
}: PerformanceChartsProps) {
  const [selectedTimeRange, setSelectedTimeRange] = useState(timeRange);
  const [performanceData, setPerformanceData] = useState<
    PerformanceDataPoint[]
  >([]);
  const [metricSummary, setMetricSummary] = useState<MetricSummary[]>([]);
  const [executionStats, setExecutionStats] = useState<ExecutionStats[]>([]);
  const [loading, setLoading] = useState(false);

  // Generate mock performance data
  const generatePerformanceData = (range: string) => {
    const dataPoints: PerformanceDataPoint[] = [];
    const now = new Date();
    let intervalMinutes = 5;
    let totalPoints = 72; // 6 hours by default

    switch (range) {
      case "1h":
        intervalMinutes = 1;
        totalPoints = 60;
        break;
      case "6h":
        intervalMinutes = 5;
        totalPoints = 72;
        break;
      case "24h":
        intervalMinutes = 20;
        totalPoints = 72;
        break;
      case "7d":
        intervalMinutes = 120;
        totalPoints = 84;
        break;
      case "30d":
        intervalMinutes = 480;
        totalPoints = 90;
        break;
    }

    for (let i = totalPoints; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * intervalMinutes * 60000);
      const timeStr =
        range === "7d" || range === "30d"
          ? timestamp.toLocaleDateString()
          : timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            });

      // Generate realistic performance data with some correlation
      const baseLoad = 40 + Math.sin((i / totalPoints) * Math.PI * 4) * 15;
      const randomVariation = (Math.random() - 0.5) * 20;

      dataPoints.push({
        timestamp: timeStr,
        cpu: Math.max(0, Math.min(100, baseLoad + randomVariation)),
        memory: Math.max(
          0,
          Math.min(100, baseLoad + 20 + randomVariation * 0.5),
        ),
        disk: Math.max(
          0,
          Math.min(100, 25 + Math.sin(i / 10) * 5 + randomVariation * 0.2),
        ),
        network: Math.max(0, baseLoad * 4 + randomVariation * 2),
        requests: Math.max(
          0,
          1000 + Math.sin(i / 8) * 400 + randomVariation * 50,
        ),
        responseTime: Math.max(
          0,
          50 + Math.sin(i / 6) * 20 + randomVariation * 0.5,
        ),
        errors: Math.max(0, Math.random() * 10),
        activeUsers: Math.max(
          0,
          200 + Math.sin(i / 12) * 80 + randomVariation * 2,
        ),
      });
    }

    return dataPoints;
  };

  // Generate metric summaries
  const generateMetricSummary = (
    data: PerformanceDataPoint[],
  ): MetricSummary[] => {
    if (data.length === 0) return [];

    const metrics = [
      { key: "cpu", name: "CPU Usage", unit: "%" },
      { key: "memory", name: "Memory Usage", unit: "%" },
      { key: "disk", name: "Disk Usage", unit: "%" },
      { key: "network", name: "Network I/O", unit: "MB/s" },
      { key: "requests", name: "Requests/min", unit: "req/min" },
      { key: "responseTime", name: "Response Time", unit: "ms" },
    ];

    return metrics.map((metric) => {
      const values = data.map(
        (d) => d[metric.key as keyof PerformanceDataPoint] as number,
      );
      const current = values[values.length - 1] || 0;
      const previous = values[values.length - 2] || current;
      const average = values.reduce((sum, val) => sum + val, 0) / values.length;
      const peak = Math.max(...values);
      const change = ((current - previous) / previous) * 100;

      let trend: "up" | "down" | "stable" = "stable";
      if (Math.abs(change) > 2) {
        trend = change > 0 ? "up" : "down";
      }

      return {
        metric: metric.name,
        current: Number(current.toFixed(1)),
        average: Number(average.toFixed(1)),
        peak: Number(peak.toFixed(1)),
        unit: metric.unit,
        trend,
        change: Number(change.toFixed(1)),
      };
    });
  };

  // Generate execution statistics
  const generateExecutionStats = (): ExecutionStats[] => [
    { name: "Completed", value: 1247, color: "#39FF14" },
    { name: "Running", value: 23, color: "#C7EA46" },
    { name: "Failed", value: 18, color: "#FF3131" },
    { name: "Pending", value: 156, color: "#00FFFF" },
    { name: "Cancelled", value: 7, color: "#FF5F1F" },
  ];

  // Load data based on time range
  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      const data = generatePerformanceData(selectedTimeRange);
      setPerformanceData(data);
      setMetricSummary(generateMetricSummary(data));
      setExecutionStats(generateExecutionStats());
      setLoading(false);
    }, 500);
  }, [selectedTimeRange]);

  // Auto-refresh data
  useEffect(() => {
    const interval = setInterval(() => {
      const data = generatePerformanceData(selectedTimeRange);
      setPerformanceData(data);
      setMetricSummary(generateMetricSummary(data));
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [selectedTimeRange, refreshInterval]);

  const timeRangeOptions = [
    { value: "1h", label: "1 Hour" },
    { value: "6h", label: "6 Hours" },
    { value: "24h", label: "24 Hours" },
    { value: "7d", label: "7 Days" },
    { value: "30d", label: "30 Days" },
  ];

  const customTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-cyber-dark border border-neon-cyan/50 p-3 rounded-lg shadow-lg">
          <p className="text-neon-cyan font-medium">{`Time: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.value}${
                entry.dataKey === "cpu" ||
                entry.dataKey === "memory" ||
                entry.dataKey === "disk"
                  ? "%"
                  : entry.dataKey === "network"
                    ? " MB/s"
                    : entry.dataKey === "responseTime"
                      ? "ms"
                      : ""
              }`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const exportData = () => {
    const csvContent = [
      Object.keys(performanceData[0] || {}).join(","),
      ...performanceData.map((row) => Object.values(row).join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `performance-data-${selectedTimeRange}-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header with Time Range Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-neon-cyan">
            Performance Analytics
          </h2>
          <p className="text-neon-cyan/70">
            System performance metrics and trends
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 border border-neon-cyan/30 rounded-lg p-1">
            {timeRangeOptions.map((option) => (
              <Button
                key={option.value}
                onClick={() => setSelectedTimeRange(option.value as any)}
                variant={
                  selectedTimeRange === option.value ? "default" : "ghost"
                }
                size="sm"
                className={cn(
                  "text-xs",
                  selectedTimeRange === option.value
                    ? "bg-neon-cyan text-black hover:bg-neon-cyan/80"
                    : "text-neon-cyan hover:bg-neon-cyan/10",
                )}
              >
                {option.label}
              </Button>
            ))}
          </div>
          <Button
            onClick={exportData}
            variant="outline"
            size="sm"
            className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Metric Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {metricSummary.map((metric) => (
          <Card
            key={metric.metric}
            className="bg-cyber-dark border-neon-cyan/30"
          >
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium text-neon-cyan/70">
                  {metric.metric}
                </h3>
                <div className="flex items-center gap-1">
                  {metric.trend === "up" && (
                    <TrendingUp className="h-4 w-4 text-neon-red" />
                  )}
                  {metric.trend === "down" && (
                    <TrendingDown className="h-4 w-4 text-neon-green" />
                  )}
                  {metric.trend === "stable" && (
                    <Activity className="h-4 w-4 text-neon-cyan" />
                  )}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xl font-bold text-white">
                  {metric.current}{" "}
                  <span className="text-sm text-neon-cyan/70">
                    {metric.unit}
                  </span>
                </div>
                <div className="flex justify-between text-xs text-neon-cyan/50">
                  <span>
                    Avg: {metric.average}
                    {metric.unit}
                  </span>
                  <span>
                    Peak: {metric.peak}
                    {metric.unit}
                  </span>
                </div>
                {Math.abs(metric.change) > 0.1 && (
                  <div
                    className={cn(
                      "text-xs",
                      metric.change > 0 ? "text-neon-red" : "text-neon-green",
                    )}
                  >
                    {metric.change > 0 ? "+" : ""}
                    {metric.change}% from previous
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Resource Utilization Chart */}
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardHeader>
            <CardTitle className="text-neon-cyan flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Resource Utilization
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ffff20" />
                <XAxis dataKey="timestamp" stroke="#00ffff70" fontSize={12} />
                <YAxis stroke="#00ffff70" fontSize={12} />
                <Tooltip content={customTooltip} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="cpu"
                  stackId="1"
                  stroke="#FF0090"
                  fill="#FF009020"
                  name="CPU %"
                />
                <Area
                  type="monotone"
                  dataKey="memory"
                  stackId="1"
                  stroke="#C7EA46"
                  fill="#C7EA4620"
                  name="Memory %"
                />
                <Area
                  type="monotone"
                  dataKey="disk"
                  stackId="1"
                  stroke="#00FFFF"
                  fill="#00FFFF20"
                  name="Disk %"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Network & Response Time */}
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardHeader>
            <CardTitle className="text-neon-cyan flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Network & Response Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ffff20" />
                <XAxis dataKey="timestamp" stroke="#00ffff70" fontSize={12} />
                <YAxis yAxisId="left" stroke="#00ffff70" fontSize={12} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#ff5f1f"
                  fontSize={12}
                />
                <Tooltip content={customTooltip} />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="network"
                  stroke="#00FFFF"
                  strokeWidth={2}
                  dot={false}
                  name="Network MB/s"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="responseTime"
                  stroke="#FF5F1F"
                  strokeWidth={2}
                  dot={false}
                  name="Response Time ms"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Request Volume */}
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardHeader>
            <CardTitle className="text-neon-cyan flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Request Volume & Users
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ffff20" />
                <XAxis dataKey="timestamp" stroke="#00ffff70" fontSize={12} />
                <YAxis yAxisId="left" stroke="#00ffff70" fontSize={12} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#ff0090"
                  fontSize={12}
                />
                <Tooltip content={customTooltip} />
                <Legend />
                <Bar
                  yAxisId="left"
                  dataKey="requests"
                  fill="#39FF14"
                  name="Requests/min"
                  opacity={0.8}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="activeUsers"
                  stroke="#FF0090"
                  strokeWidth={2}
                  dot={false}
                  name="Active Users"
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Execution Statistics */}
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardHeader>
            <CardTitle className="text-neon-cyan flex items-center gap-2">
              <PieChartIcon className="h-5 w-5" />
              Execution Status Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={executionStats}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {executionStats.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0a0a0a",
                    border: "1px solid #00ffff50",
                    borderRadius: "8px",
                    color: "#ffffff",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Error Rate Trends */}
      <Card className="bg-cyber-dark border-neon-cyan/30">
        <CardHeader>
          <CardTitle className="text-neon-cyan flex items-center gap-2">
            <Server className="h-5 w-5" />
            Error Rate Trends
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00ffff20" />
              <XAxis dataKey="timestamp" stroke="#00ffff70" fontSize={12} />
              <YAxis stroke="#00ffff70" fontSize={12} />
              <Tooltip content={customTooltip} />
              <Line
                type="monotone"
                dataKey="errors"
                stroke="#FF3131"
                strokeWidth={3}
                dot={{ fill: "#FF3131", strokeWidth: 2, r: 4 }}
                name="Errors/min"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {loading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-neon-cyan"></div>
        </div>
      )}
    </div>
  );
}

export default PerformanceCharts;
