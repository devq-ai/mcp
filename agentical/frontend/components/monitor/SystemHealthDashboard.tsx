"use client";

import * as React from "react";
import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Server,
  Database,
  Cpu,
  HardDrive,
  Wifi,
  HardDrive as MemoryIcon,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Bell,
  Shield,
  Users,
  Globe,
  Monitor,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SystemHealthMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  status: "healthy" | "warning" | "critical" | "unknown";
  trend: "up" | "down" | "stable";
  threshold: {
    warning: number;
    critical: number;
  };
  lastUpdated: string;
}

interface ServiceHealth {
  id: string;
  name: string;
  status: "online" | "offline" | "degraded" | "maintenance";
  uptime: number;
  responseTime: number;
  lastCheck: string;
  endpoint?: string;
  version?: string;
}

interface SystemAlert {
  id: string;
  severity: "low" | "medium" | "high" | "critical";
  title: string;
  message: string;
  timestamp: string;
  category: "performance" | "security" | "availability" | "capacity";
  source: string;
  acknowledged: boolean;
}

interface SystemHealthDashboardProps {
  refreshInterval?: number;
  className?: string;
  compact?: boolean;
}

export function SystemHealthDashboard({
  refreshInterval = 30000,
  className,
  compact = false,
}: SystemHealthDashboardProps) {
  const [systemMetrics, setSystemMetrics] = useState<SystemHealthMetric[]>([
    {
      id: "cpu",
      name: "CPU Usage",
      value: 45.2,
      unit: "%",
      status: "healthy",
      trend: "stable",
      threshold: { warning: 70, critical: 85 },
      lastUpdated: new Date().toISOString(),
    },
    {
      id: "memory",
      name: "Memory Usage",
      value: 68.5,
      unit: "%",
      status: "warning",
      trend: "up",
      threshold: { warning: 80, critical: 90 },
      lastUpdated: new Date().toISOString(),
    },
    {
      id: "disk",
      name: "Disk Usage",
      value: 23.7,
      unit: "%",
      status: "healthy",
      trend: "stable",
      threshold: { warning: 75, critical: 90 },
      lastUpdated: new Date().toISOString(),
    },
    {
      id: "network",
      name: "Network I/O",
      value: 156.8,
      unit: "MB/s",
      status: "healthy",
      trend: "up",
      threshold: { warning: 800, critical: 950 },
      lastUpdated: new Date().toISOString(),
    },
  ]);

  const [services] = useState<ServiceHealth[]>([
    {
      id: "api",
      name: "API Server",
      status: "online",
      uptime: 99.98,
      responseTime: 45,
      lastCheck: new Date().toISOString(),
      endpoint: "https://api.agentical.com/health",
      version: "v1.2.3",
    },
    {
      id: "database",
      name: "SurrealDB",
      status: "online",
      uptime: 99.95,
      responseTime: 12,
      lastCheck: new Date().toISOString(),
      endpoint: "ws://localhost:8000/rpc",
      version: "1.0.0",
    },
    {
      id: "redis",
      name: "Redis Cache",
      status: "online",
      uptime: 100.0,
      responseTime: 2,
      lastCheck: new Date().toISOString(),
      endpoint: "redis://localhost:6379",
      version: "7.0.5",
    },
    {
      id: "logfire",
      name: "Logfire Observability",
      status: "online",
      uptime: 99.99,
      responseTime: 120,
      lastCheck: new Date().toISOString(),
      endpoint: "https://logfire-us.pydantic.dev",
      version: "0.28.0",
    },
    {
      id: "scheduler",
      name: "Task Scheduler",
      status: "degraded",
      uptime: 98.5,
      responseTime: 180,
      lastCheck: new Date().toISOString(),
      version: "1.0.0",
    },
  ]);

  const [alerts, setAlerts] = useState<SystemAlert[]>([
    {
      id: "alert-1",
      severity: "medium",
      title: "High Memory Usage",
      message: "Memory usage has exceeded 65% threshold",
      timestamp: new Date(Date.now() - 300000).toISOString(),
      category: "performance",
      source: "System Monitor",
      acknowledged: false,
    },
    {
      id: "alert-2",
      severity: "low",
      title: "Task Scheduler Degraded",
      message: "Task scheduler response time increased to 180ms",
      timestamp: new Date(Date.now() - 600000).toISOString(),
      category: "availability",
      source: "Health Check",
      acknowledged: false,
    },
    {
      id: "alert-3",
      severity: "high",
      title: "Security Scan Alert",
      message: "Unusual API access patterns detected",
      timestamp: new Date(Date.now() - 900000).toISOString(),
      category: "security",
      source: "Security Monitor",
      acknowledged: true,
    },
  ]);

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Real-time updates simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemMetrics((prev) =>
        prev.map((metric) => {
          let newValue = metric.value;
          let newStatus = metric.status;
          let newTrend = metric.trend;

          // Simulate realistic changes
          switch (metric.id) {
            case "cpu":
              newValue = Math.max(
                0,
                Math.min(100, metric.value + (Math.random() - 0.5) * 8),
              );
              break;
            case "memory":
              newValue = Math.max(
                0,
                Math.min(100, metric.value + (Math.random() - 0.5) * 3),
              );
              break;
            case "disk":
              newValue = Math.max(
                0,
                Math.min(100, metric.value + (Math.random() - 0.5) * 1),
              );
              break;
            case "network":
              newValue = Math.max(0, metric.value + (Math.random() - 0.5) * 20);
              break;
          }

          // Update status based on thresholds
          if (newValue >= metric.threshold.critical) {
            newStatus = "critical";
          } else if (newValue >= metric.threshold.warning) {
            newStatus = "warning";
          } else {
            newStatus = "healthy";
          }

          // Update trend
          if (newValue > metric.value + 1) {
            newTrend = "up";
          } else if (newValue < metric.value - 1) {
            newTrend = "down";
          } else {
            newTrend = "stable";
          }

          return {
            ...metric,
            value: Number(newValue.toFixed(1)),
            status: newStatus,
            trend: newTrend,
            lastUpdated: new Date().toISOString(),
          };
        }),
      );

      setLastUpdated(new Date());
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsRefreshing(false);
    setLastUpdated(new Date());
  }, []);

  const acknowledgeAlert = useCallback((alertId: string) => {
    setAlerts((prev) =>
      prev.map((alert) =>
        alert.id === alertId ? { ...alert, acknowledged: true } : alert,
      ),
    );
  }, []);

  const getMetricIcon = (id: string) => {
    switch (id) {
      case "cpu":
        return Cpu;
      case "memory":
        return MemoryIcon;
      case "disk":
        return HardDrive;
      case "network":
        return Wifi;
      default:
        return Activity;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
      case "online":
        return "text-neon-green border-neon-green/30 bg-neon-green/10";
      case "warning":
      case "degraded":
        return "text-neon-yellow border-neon-yellow/30 bg-neon-yellow/10";
      case "critical":
      case "offline":
        return "text-neon-red border-neon-red/30 bg-neon-red/10";
      case "maintenance":
        return "text-neon-cyan border-neon-cyan/30 bg-neon-cyan/10";
      default:
        return "text-gray-400 border-gray-400/30 bg-gray-400/10";
    }
  };

  const getServiceIcon = (id: string) => {
    switch (id) {
      case "api":
        return Globe;
      case "database":
        return Database;
      case "redis":
        return Server;
      case "logfire":
        return Monitor;
      case "scheduler":
        return Clock;
      default:
        return Server;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up":
        return TrendingUp;
      case "down":
        return TrendingDown;
      default:
        return Minus;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "text-neon-red bg-neon-red/20 border-neon-red/50";
      case "high":
        return "text-neon-orange bg-neon-orange/20 border-neon-orange/50";
      case "medium":
        return "text-neon-yellow bg-neon-yellow/20 border-neon-yellow/50";
      case "low":
        return "text-neon-cyan bg-neon-cyan/20 border-neon-cyan/50";
      default:
        return "text-gray-400 bg-gray-400/20 border-gray-400/50";
    }
  };

  const overallHealth = React.useMemo(() => {
    const criticalCount = systemMetrics.filter(
      (m) => m.status === "critical",
    ).length;
    const warningCount = systemMetrics.filter(
      (m) => m.status === "warning",
    ).length;
    const offlineServices = services.filter(
      (s) => s.status === "offline",
    ).length;

    if (criticalCount > 0 || offlineServices > 0) return "critical";
    if (warningCount > 0) return "warning";
    return "healthy";
  }, [systemMetrics, services]);

  const activeAlerts = alerts.filter((alert) => !alert.acknowledged);

  if (compact) {
    return (
      <div className={cn("space-y-4", className)}>
        {/* Compact Overview */}
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div
                  className={cn(
                    "w-3 h-3 rounded-full",
                    overallHealth === "healthy"
                      ? "bg-neon-green"
                      : overallHealth === "warning"
                        ? "bg-neon-yellow"
                        : "bg-neon-red",
                  )}
                />
                <span className="text-white font-medium">System Health</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-neon-cyan/70">
                <Bell className="h-4 w-4" />
                <span>{activeAlerts.length} alerts</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-neon-cyan">
            System Health Dashboard
          </h2>
          <p className="text-neon-cyan/70">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge className={getStatusColor(overallHealth)}>
            {overallHealth === "healthy" && (
              <CheckCircle className="h-4 w-4 mr-1" />
            )}
            {overallHealth === "warning" && (
              <AlertTriangle className="h-4 w-4 mr-1" />
            )}
            {overallHealth === "critical" && (
              <AlertTriangle className="h-4 w-4 mr-1" />
            )}
            {overallHealth.toUpperCase()}
          </Badge>
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

      {/* System Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {systemMetrics.map((metric) => {
          const IconComponent = getMetricIcon(metric.id);
          const TrendIcon = getTrendIcon(metric.trend);

          return (
            <Card key={metric.id} className="bg-cyber-dark border-neon-cyan/30">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <IconComponent className="h-5 w-5 text-neon-cyan" />
                    <span className="text-sm font-medium text-white">
                      {metric.name}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <TrendIcon
                      className={cn(
                        "h-4 w-4",
                        metric.trend === "up"
                          ? "text-neon-red"
                          : metric.trend === "down"
                            ? "text-neon-green"
                            : "text-neon-cyan",
                      )}
                    />
                    <Badge
                      className={getStatusColor(metric.status)}
                      variant="outline"
                    >
                      {metric.status}
                    </Badge>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-baseline gap-1">
                    <span className="text-2xl font-bold text-white">
                      {metric.value}
                    </span>
                    <span className="text-sm text-neon-cyan/70">
                      {metric.unit}
                    </span>
                  </div>

                  <Progress
                    value={
                      metric.id === "network"
                        ? Math.min((metric.value / 1000) * 100, 100)
                        : metric.value
                    }
                    className="h-2"
                  />

                  <div className="flex justify-between text-xs text-neon-cyan/50">
                    <span>
                      Warning: {metric.threshold.warning}
                      {metric.unit}
                    </span>
                    <span>
                      Critical: {metric.threshold.critical}
                      {metric.unit}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Services Status */}
      <Card className="bg-cyber-dark border-neon-cyan/30">
        <CardHeader>
          <CardTitle className="text-neon-cyan flex items-center gap-2">
            <Server className="h-5 w-5" />
            Service Health
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {services.map((service) => {
              const ServiceIcon = getServiceIcon(service.id);

              return (
                <div
                  key={service.id}
                  className="p-4 rounded-lg border border-neon-cyan/20 bg-neon-cyan/5"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <ServiceIcon className="h-4 w-4 text-neon-cyan" />
                      <span className="font-medium text-white">
                        {service.name}
                      </span>
                    </div>
                    <Badge
                      className={getStatusColor(service.status)}
                      variant="outline"
                    >
                      {service.status}
                    </Badge>
                  </div>

                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-neon-cyan/70">Uptime:</span>
                      <span className="text-white">{service.uptime}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neon-cyan/70">Response:</span>
                      <span className="text-white">
                        {service.responseTime}ms
                      </span>
                    </div>
                    {service.version && (
                      <div className="flex justify-between">
                        <span className="text-neon-cyan/70">Version:</span>
                        <span className="text-white">{service.version}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Active Alerts */}
      {activeAlerts.length > 0 && (
        <Card className="bg-cyber-dark border-neon-red/30">
          <CardHeader>
            <CardTitle className="text-neon-red flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Active Alerts ({activeAlerts.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {activeAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className="p-3 rounded-lg border border-neon-red/20 bg-neon-red/5"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge
                          className={getSeverityColor(alert.severity)}
                          variant="outline"
                        >
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <span className="text-sm text-neon-cyan/70">
                          {alert.category}
                        </span>
                      </div>
                      <h4 className="font-medium text-white mb-1">
                        {alert.title}
                      </h4>
                      <p className="text-sm text-neon-cyan/70 mb-2">
                        {alert.message}
                      </p>
                      <div className="flex items-center gap-4 text-xs text-neon-cyan/50">
                        <span>Source: {alert.source}</span>
                        <span>
                          Time: {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                    <Button
                      onClick={() => acknowledgeAlert(alert.id)}
                      variant="outline"
                      size="sm"
                      className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
                    >
                      Acknowledge
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <Users className="h-8 w-8 text-neon-cyan mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">247</div>
            <div className="text-sm text-neon-cyan/70">Active Users</div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <Zap className="h-8 w-8 text-neon-lime mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">1,342</div>
            <div className="text-sm text-neon-cyan/70">API Requests/min</div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <Shield className="h-8 w-8 text-neon-green mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">99.97%</div>
            <div className="text-sm text-neon-cyan/70">Security Score</div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <Activity className="h-8 w-8 text-neon-magenta mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">23</div>
            <div className="text-sm text-neon-cyan/70">Active Workflows</div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default SystemHealthDashboard;
