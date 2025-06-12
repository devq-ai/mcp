"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  AlertTriangle,
  XCircle,
  CheckCircle,
  Info,
  X,
  Bell,
  BellOff,
  Filter,
  Download,
  Trash2,
  Eye,
  EyeOff,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Alert {
  id: string;
  type: "critical" | "warning" | "info" | "success";
  title: string;
  message: string;
  timestamp: Date;
  source: string;
  executionId?: string;
  playbookName?: string;
  isRead: boolean;
  isResolved: boolean;
  severity: 1 | 2 | 3 | 4 | 5; // 1 = lowest, 5 = highest
  category: "system" | "execution" | "performance" | "security" | "network";
  data?: Record<string, any>;
}

interface AlertRule {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  enabled: boolean;
  category: Alert["category"];
  severity: Alert["severity"];
}

interface AlertSystemProps {
  maxAlerts?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
  className?: string;
}

export default function AlertSystem({
  maxAlerts = 100,
  autoRefresh = true,
  refreshInterval = 5000,
  className,
}: AlertSystemProps) {
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: "alert-001",
      type: "critical",
      title: "High Memory Usage",
      message: "System memory usage has exceeded 90% threshold",
      timestamp: new Date(Date.now() - 120000),
      source: "system-monitor",
      isRead: false,
      isResolved: false,
      severity: 5,
      category: "system",
      data: { currentUsage: 92.5, threshold: 90 },
    },
    {
      id: "alert-002",
      type: "warning",
      title: "Execution Timeout",
      message: "Playbook execution is taking longer than expected",
      timestamp: new Date(Date.now() - 300000),
      source: "execution-engine",
      executionId: "exec-001",
      playbookName: "User Onboarding Flow",
      isRead: true,
      isResolved: false,
      severity: 3,
      category: "execution",
      data: { expectedTime: 60, actualTime: 180 },
    },
    {
      id: "alert-003",
      type: "info",
      title: "Performance Improvement",
      message: "System throughput has increased by 15%",
      timestamp: new Date(Date.now() - 600000),
      source: "performance-monitor",
      isRead: true,
      isResolved: true,
      severity: 1,
      category: "performance",
      data: { improvement: 15.3 },
    },
    {
      id: "alert-004",
      type: "warning",
      title: "Failed API Calls",
      message: "Multiple API calls have failed in the last 5 minutes",
      timestamp: new Date(Date.now() - 900000),
      source: "api-monitor",
      isRead: false,
      isResolved: false,
      severity: 4,
      category: "network",
      data: { failedCalls: 12, successRate: 78.5 },
    },
    {
      id: "alert-005",
      type: "success",
      title: "Security Scan Complete",
      message: "Weekly security scan completed successfully with no issues",
      timestamp: new Date(Date.now() - 1800000),
      source: "security-scanner",
      isRead: true,
      isResolved: true,
      severity: 2,
      category: "security",
      data: { scannedItems: 1250, issues: 0 },
    },
  ]);

  const [alertRules, setAlertRules] = useState<AlertRule[]>([
    {
      id: "rule-001",
      name: "High CPU Usage",
      condition: "cpu_usage > threshold",
      threshold: 85,
      enabled: true,
      category: "system",
      severity: 4,
    },
    {
      id: "rule-002",
      name: "Execution Failure Rate",
      condition: "failure_rate > threshold",
      threshold: 5,
      enabled: true,
      category: "execution",
      severity: 5,
    },
    {
      id: "rule-003",
      name: "Response Time Degradation",
      condition: "avg_response_time > threshold",
      threshold: 2000,
      enabled: true,
      category: "performance",
      severity: 3,
    },
  ]);

  const [filter, setFilter] = useState<{
    type: Alert["type"] | "all";
    category: Alert["category"] | "all";
    isRead: boolean | "all";
    isResolved: boolean | "all";
  }>({
    type: "all",
    category: "all",
    isRead: "all",
    isResolved: "all",
  });

  const [isNotificationsEnabled, setIsNotificationsEnabled] = useState(true);
  const [, setSelectedAlert] = useState<Alert | null>(null);
  const [sortBy, setSortBy] = useState<"timestamp" | "severity">("timestamp");

  // Simulate real-time alerts
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Randomly generate new alerts
      if (Math.random() < 0.1) {
        const alertTypes: Alert["type"][] = [
          "critical",
          "warning",
          "info",
          "success",
        ];
        const categories: Alert["category"][] = [
          "system",
          "execution",
          "performance",
          "security",
          "network",
        ];

        const messages = {
          critical: [
            "Database connection lost",
            "Disk space critically low",
            "Service unavailable",
            "Security breach detected",
          ],
          warning: [
            "High error rate detected",
            "Slow response times",
            "Memory usage above threshold",
            "Backup process delayed",
          ],
          info: [
            "System maintenance scheduled",
            "New version available",
            "Health check completed",
            "Configuration updated",
          ],
          success: [
            "Backup completed successfully",
            "Performance optimization applied",
            "Security patch installed",
            "System health restored",
          ],
        };

        const randomType =
          alertTypes[Math.floor(Math.random() * alertTypes.length)]!;
        const randomCategory =
          categories[Math.floor(Math.random() * categories.length)]!;
        const randomMessages = messages[randomType];
        const randomMessage =
          randomMessages[Math.floor(Math.random() * randomMessages.length)] ||
          "Alert message";

        const newAlert: Alert = {
          id: `alert-${Date.now()}`,
          type: randomType,
          title: randomMessage,
          message: `${randomMessage} - automatic detection`,
          timestamp: new Date(),
          source: "auto-monitor",
          isRead: false,
          isResolved: false,
          severity: (Math.floor(Math.random() * 5) + 1) as Alert["severity"],
          category: randomCategory,
        };

        setAlerts((prev) => [newAlert, ...prev].slice(0, maxAlerts));

        // Show browser notification if enabled
        if (
          isNotificationsEnabled &&
          "Notification" in window &&
          Notification.permission === "granted"
        ) {
          new Notification(
            `${newAlert.type.toUpperCase()}: ${newAlert.title}`,
            {
              body: newAlert.message,
              icon: "/favicon.ico",
            },
          );
        }
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, maxAlerts, isNotificationsEnabled]);

  // Request notification permission
  useEffect(() => {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  const getAlertIcon = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-neon-yellow" />;
      case "success":
        return <CheckCircle className="h-4 w-4 text-neon-green" />;
      default:
        return <Info className="h-4 w-4 text-neon-cyan" />;
    }
  };

  const getAlertColor = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
        return "text-neon-red border-neon-red/50 bg-neon-red/10";
      case "warning":
        return "text-neon-yellow border-neon-yellow/50 bg-neon-yellow/10";
      case "success":
        return "text-neon-green border-neon-green/50 bg-neon-green/10";
      default:
        return "text-neon-cyan border-neon-cyan/50 bg-neon-cyan/10";
    }
  };

  const getSeverityColor = (severity: Alert["severity"]) => {
    switch (severity) {
      case 5:
        return "bg-neon-red";
      case 4:
        return "bg-neon-orange";
      case 3:
        return "bg-neon-yellow";
      case 2:
        return "bg-neon-cyan";
      default:
        return "bg-neon-green";
    }
  };

  const filteredAlerts = alerts
    .filter((alert) => {
      if (filter.type !== "all" && alert.type !== filter.type) return false;
      if (filter.category !== "all" && alert.category !== filter.category)
        return false;
      if (filter.isRead !== "all" && alert.isRead !== filter.isRead)
        return false;
      if (filter.isResolved !== "all" && alert.isResolved !== filter.isResolved)
        return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "severity") {
        return b.severity - a.severity;
      }
      return b.timestamp.getTime() - a.timestamp.getTime();
    });

  const unreadCount = alerts.filter((alert) => !alert.isRead).length;
  const criticalCount = alerts.filter(
    (alert) => alert.type === "critical" && !alert.isResolved,
  ).length;

  const markAsRead = (alertId: string) => {
    setAlerts((prev) =>
      prev.map((alert) =>
        alert.id === alertId ? { ...alert, isRead: true } : alert,
      ),
    );
  };

  const markAsResolved = (alertId: string) => {
    setAlerts((prev) =>
      prev.map((alert) =>
        alert.id === alertId
          ? { ...alert, isResolved: true, isRead: true }
          : alert,
      ),
    );
  };

  const deleteAlert = (alertId: string) => {
    setAlerts((prev) => prev.filter((alert) => alert.id !== alertId));
  };

  const markAllAsRead = () => {
    setAlerts((prev) => prev.map((alert) => ({ ...alert, isRead: true })));
  };

  const clearResolved = () => {
    setAlerts((prev) => prev.filter((alert) => !alert.isResolved));
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl font-bold text-neon-cyan">Alert System</h2>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={cn(
                "text-xs",
                unreadCount > 0
                  ? "text-neon-red border-neon-red/50 bg-neon-red/10"
                  : "text-neon-green border-neon-green/50 bg-neon-green/10",
              )}
            >
              {unreadCount} unread
            </Badge>
            {criticalCount > 0 && (
              <Badge
                variant="outline"
                className="text-xs text-neon-red border-neon-red/50 bg-neon-red/10"
              >
                {criticalCount} critical
              </Badge>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setIsNotificationsEnabled(!isNotificationsEnabled)}
            className={cn(
              "border-neon-cyan/50",
              isNotificationsEnabled ? "text-neon-green" : "text-neon-red",
            )}
          >
            {isNotificationsEnabled ? (
              <Bell className="h-4 w-4" />
            ) : (
              <BellOff className="h-4 w-4" />
            )}
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={markAllAsRead}
            disabled={unreadCount === 0}
            className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
          >
            <Eye className="h-4 w-4 mr-1" />
            Mark All Read
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={clearResolved}
            className="border-neon-green/50 text-neon-green hover:bg-neon-green/10"
          >
            <Trash2 className="h-4 w-4 mr-1" />
            Clear Resolved
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card className="bg-cyber-dark border-neon-cyan/20">
        <CardContent className="p-4">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-neon-cyan" />
              <span className="text-sm text-neon-cyan">Filters:</span>
            </div>

            <select
              value={filter.type}
              onChange={(e) =>
                setFilter((prev) => ({ ...prev, type: e.target.value as any }))
              }
              className="bg-cyber-dark border border-neon-cyan/30 text-neon-cyan text-sm rounded px-2 py-1"
            >
              <option value="all">All Types</option>
              <option value="critical">Critical</option>
              <option value="warning">Warning</option>
              <option value="info">Info</option>
              <option value="success">Success</option>
            </select>

            <select
              value={filter.category}
              onChange={(e) =>
                setFilter((prev) => ({
                  ...prev,
                  category: e.target.value as any,
                }))
              }
              className="bg-cyber-dark border border-neon-cyan/30 text-neon-cyan text-sm rounded px-2 py-1"
            >
              <option value="all">All Categories</option>
              <option value="system">System</option>
              <option value="execution">Execution</option>
              <option value="performance">Performance</option>
              <option value="security">Security</option>
              <option value="network">Network</option>
            </select>

            <select
              value={filter.isRead.toString()}
              onChange={(e) =>
                setFilter((prev) => ({
                  ...prev,
                  isRead:
                    e.target.value === "all"
                      ? "all"
                      : e.target.value === "true",
                }))
              }
              className="bg-cyber-dark border border-neon-cyan/30 text-neon-cyan text-sm rounded px-2 py-1"
            >
              <option value="all">All Status</option>
              <option value="false">Unread</option>
              <option value="true">Read</option>
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-cyber-dark border border-neon-cyan/30 text-neon-cyan text-sm rounded px-2 py-1"
            >
              <option value="timestamp">Sort by Time</option>
              <option value="severity">Sort by Severity</option>
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Alerts List */}
      <Card className="bg-cyber-dark border-neon-cyan/20">
        <CardHeader>
          <CardTitle className="text-neon-cyan flex items-center justify-between">
            <span>Active Alerts ({filteredAlerts.length})</span>
            <Button size="sm" variant="outline" className="border-neon-cyan/50">
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-96">
            <div className="space-y-3">
              {filteredAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className={cn(
                    "p-4 border rounded-lg transition-all duration-200 cursor-pointer",
                    alert.isRead ? "bg-cyber-dark/30" : "bg-cyber-dark/60",
                    alert.isResolved ? "opacity-60" : "",
                    getAlertColor(alert.type),
                  )}
                  onClick={() => setSelectedAlert(alert)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      {getAlertIcon(alert.type)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3
                            className={cn(
                              "font-semibold text-sm",
                              alert.isRead ? "text-white/80" : "text-white",
                            )}
                          >
                            {alert.title}
                          </h3>
                          <div
                            className={cn(
                              "w-2 h-2 rounded-full",
                              getSeverityColor(alert.severity),
                            )}
                          />
                          <Badge
                            variant="outline"
                            className={cn("text-xs", getAlertColor(alert.type))}
                          >
                            {alert.category}
                          </Badge>
                          {alert.isResolved && (
                            <Badge
                              variant="outline"
                              className="text-xs text-neon-green border-neon-green/50"
                            >
                              RESOLVED
                            </Badge>
                          )}
                        </div>
                        <p
                          className={cn(
                            "text-sm mb-2",
                            alert.isRead ? "text-white/60" : "text-white/80",
                          )}
                        >
                          {alert.message}
                        </p>
                        <div className="flex items-center gap-4 text-xs text-white/50">
                          <span>{alert.timestamp.toLocaleString()}</span>
                          <span>Source: {alert.source}</span>
                          {alert.executionId && (
                            <span>Execution: {alert.executionId}</span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 ml-4">
                      {!alert.isRead && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            markAsRead(alert.id);
                          }}
                          className="h-6 w-6 p-0 text-neon-cyan hover:bg-neon-cyan/10"
                        >
                          <Eye className="h-3 w-3" />
                        </Button>
                      )}
                      {!alert.isResolved && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            markAsResolved(alert.id);
                          }}
                          className="h-6 w-6 p-0 text-neon-green hover:bg-neon-green/10"
                        >
                          <CheckCircle className="h-3 w-3" />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteAlert(alert.id);
                        }}
                        className="h-6 w-6 p-0 text-neon-red hover:bg-neon-red/10"
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>

                  {alert.data && Object.keys(alert.data).length > 0 && (
                    <div className="mt-3 p-2 bg-cyber-dark/30 border border-neon-cyan/10 rounded">
                      <pre className="text-xs text-neon-cyan/70 overflow-auto">
                        {JSON.stringify(alert.data, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              ))}

              {filteredAlerts.length === 0 && (
                <div className="text-center py-8">
                  <CheckCircle className="h-12 w-12 text-neon-green/50 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-neon-green mb-2">
                    No Alerts Found
                  </h3>
                  <p className="text-neon-cyan/70">
                    {filter.type === "all" && filter.category === "all"
                      ? "All systems are running smoothly!"
                      : "No alerts match the current filters."}
                  </p>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Alert Rules */}
      <Card className="bg-cyber-dark border-neon-cyan/20">
        <CardHeader>
          <CardTitle className="text-neon-cyan">Alert Rules</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {alertRules.map((rule) => (
              <div
                key={rule.id}
                className="flex items-center justify-between p-3 bg-cyber-dark/50 border border-neon-cyan/10 rounded"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={cn(
                      "w-2 h-2 rounded-full",
                      getSeverityColor(rule.severity),
                    )}
                  />
                  <div>
                    <h4 className="font-semibold text-white text-sm">
                      {rule.name}
                    </h4>
                    <p className="text-xs text-neon-cyan/70">
                      {rule.condition} (threshold: {rule.threshold})
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-xs",
                      rule.enabled
                        ? "text-neon-green border-neon-green/50"
                        : "text-neon-red border-neon-red/50",
                    )}
                  >
                    {rule.enabled ? "ENABLED" : "DISABLED"}
                  </Badge>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setAlertRules((prev) =>
                        prev.map((r) =>
                          r.id === rule.id ? { ...r, enabled: !r.enabled } : r,
                        ),
                      );
                    }}
                    className="h-6 w-6 p-0 text-neon-cyan hover:bg-neon-cyan/10"
                  >
                    {rule.enabled ? (
                      <EyeOff className="h-3 w-3" />
                    ) : (
                      <Eye className="h-3 w-3" />
                    )}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
