"use client";

import * as React from "react";
import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  AlertTriangle,
  XCircle,
  Info,
  Search,
  Filter,
  Clock,
  Code,
  Server,
  Database,
  Globe,
  Zap,
  RefreshCw,
  Download,
  ChevronDown,
  ChevronUp,
  Copy,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface ErrorLog {
  id: string;
  timestamp: string;
  level: "error" | "warning" | "info" | "debug";
  message: string;
  source: string;
  category:
    | "system"
    | "api"
    | "database"
    | "workflow"
    | "security"
    | "performance";
  userId?: string;
  sessionId?: string;
  requestId?: string;
  stackTrace?: string;
  context?: Record<string, any>;
  resolved: boolean;
  assignedTo?: string;
  tags: string[];
  occurrences: number;
  firstSeen: string;
  lastSeen: string;
}

interface ErrorMetrics {
  totalErrors: number;
  errorRate: number;
  criticalErrors: number;
  resolvedErrors: number;
  averageResolutionTime: number;
  topErrorSources: Array<{ source: string; count: number }>;
  errorTrends: Array<{ timestamp: string; count: number }>;
}

interface ErrorFilter {
  level: string[];
  category: string[];
  source: string[];
  resolved: boolean | null;
  timeRange: string;
  searchTerm: string;
}

interface ErrorTrackingProps {
  refreshInterval?: number;
  className?: string;
  maxErrors?: number;
}

export function ErrorTracking({
  refreshInterval = 30000,
  className,
  maxErrors = 50,
}: ErrorTrackingProps) {
  const [errors, setErrors] = useState<ErrorLog[]>([]);
  const [metrics, setMetrics] = useState<ErrorMetrics>({
    totalErrors: 0,
    errorRate: 0,
    criticalErrors: 0,
    resolvedErrors: 0,
    averageResolutionTime: 0,
    topErrorSources: [],
    errorTrends: [],
  });
  const [filter, setFilter] = useState<ErrorFilter>({
    level: [],
    category: [],
    source: [],
    resolved: null,
    timeRange: "24h",
    searchTerm: "",
  });

  const [expandedErrors, setExpandedErrors] = useState<Set<string>>(new Set());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Generate mock error data
  const generateMockErrors = (): ErrorLog[] => {
    const sources = [
      "FastAPI Server",
      "SurrealDB",
      "Redis Cache",
      "TaskMaster AI",
      "Workflow Engine",
      "Authentication Service",
      "Logfire Observer",
      "MCP Server",
    ];

    const categories: ErrorLog["category"][] = [
      "system",
      "api",
      "database",
      "workflow",
      "security",
      "performance",
    ];

    const levels: ErrorLog["level"][] = ["error", "warning", "info", "debug"];

    const mockErrors: ErrorLog[] = [];

    for (let i = 0; i < 25; i++) {
      const timestamp = new Date(
        Date.now() - Math.random() * 24 * 60 * 60 * 1000,
      );
      const source =
        sources[Math.floor(Math.random() * sources.length)] || "Unknown Source";
      const category =
        categories[Math.floor(Math.random() * categories.length)] || "system";
      const level =
        (levels[
          Math.floor(Math.random() * levels.length)
        ] as ErrorLog["level"]) || "error";
      const occurrences = Math.floor(Math.random() * 10) + 1;

      const errorMessages = {
        error: [
          "Database connection timeout",
          "API rate limit exceeded",
          "Authentication token expired",
          "Workflow execution failed",
          "Memory allocation error",
          "Network connectivity lost",
        ],
        warning: [
          "High memory usage detected",
          "Slow API response time",
          "Cache miss rate increased",
          "Deprecated API usage",
          "Configuration mismatch",
          "Resource usage threshold exceeded",
        ],
        info: [
          "User session started",
          "Workflow completed successfully",
          "Cache refreshed",
          "Configuration updated",
          "Health check passed",
          "Backup completed",
        ],
        debug: [
          "Debug: Variable state changed",
          "Debug: Function entry",
          "Debug: API request details",
          "Debug: Cache operation",
          "Debug: Database query",
          "Debug: Authentication check",
        ],
      };

      const levelMessages = errorMessages[level as keyof typeof errorMessages];
      const message =
        levelMessages[Math.floor(Math.random() * levelMessages.length)] ||
        "Unknown error";

      const baseError: ErrorLog = {
        id: `error-${i + 1}`,
        timestamp: timestamp.toISOString(),
        level,
        message,
        source,
        category,
        sessionId: `session-${Math.random().toString(36).substr(2, 9)}`,
        requestId: `req-${Math.random().toString(36).substr(2, 9)}`,
        context: {
          endpoint: `/api/v1/${["users", "workflows", "agents", "playbooks"][Math.floor(Math.random() * 4)]}`,
          method: ["GET", "POST", "PUT", "DELETE"][
            Math.floor(Math.random() * 4)
          ],
          userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
          ip: `192.168.1.${Math.floor(Math.random() * 255)}`,
        },
        resolved: Math.random() > 0.7,
        tags: ["production", category],
        occurrences,
        firstSeen: new Date(
          timestamp.getTime() - occurrences * 60000,
        ).toISOString(),
        lastSeen: timestamp.toISOString(),
      };

      // Add optional properties conditionally
      if (Math.random() > 0.7) {
        baseError.userId = `user-${Math.floor(Math.random() * 1000)}`;
      }
      if (level === "error") {
        baseError.stackTrace = generateStackTrace();
      }
      if (Math.random() > 0.8) {
        baseError.assignedTo = "dev-team";
      }

      mockErrors.push(baseError);
    }

    return mockErrors.sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );
  };

  const generateStackTrace = (): string => {
    return `Traceback (most recent call last):
  File "/app/main.py", line 45, in process_request
    result = await workflow_engine.execute(playbook_id)
  File "/app/workflows/engine.py", line 128, in execute
    step_result = await self.execute_step(step)
  File "/app/workflows/engine.py", line 156, in execute_step
    return await agent.run(step_config)
  File "/app/agents/base_agent.py", line 89, in run
    response = await self.api_client.post(endpoint, data)
ConnectionError: Unable to connect to database server`;
  };

  const generateMetrics = (errorList: ErrorLog[]): ErrorMetrics => {
    const totalErrors = errorList.length;
    const criticalErrors = errorList.filter((e) => e.level === "error").length;
    const resolvedErrors = errorList.filter((e) => e.resolved).length;
    const errorRate =
      totalErrors > 0 ? (criticalErrors / totalErrors) * 100 : 0;

    const sourceCounts = errorList.reduce(
      (acc, error) => {
        acc[error.source] = (acc[error.source] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );

    const topErrorSources = Object.entries(sourceCounts)
      .map(([source, count]) => ({ source, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    const now = new Date();
    const errorTrends = Array.from({ length: 24 }, (_, i) => {
      const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
      const hourStart = new Date(hour.setMinutes(0, 0, 0));
      const hourEnd = new Date(hourStart.getTime() + 60 * 60 * 1000);

      const count = errorList.filter((error) => {
        const errorTime = new Date(error.timestamp);
        return errorTime >= hourStart && errorTime < hourEnd;
      }).length;

      return {
        timestamp: hourStart.toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
        count,
      };
    }).reverse();

    return {
      totalErrors,
      errorRate: Number(errorRate.toFixed(1)),
      criticalErrors,
      resolvedErrors,
      averageResolutionTime: 2.5, // hours
      topErrorSources,
      errorTrends,
    };
  };

  // Initialize data
  useEffect(() => {
    const mockErrors = generateMockErrors();
    setErrors(mockErrors);
    setMetrics(generateMetrics(mockErrors));
  }, []);

  // Auto-refresh
  useEffect(() => {
    const interval = setInterval(() => {
      const mockErrors = generateMockErrors();
      setErrors(mockErrors);
      setMetrics(generateMetrics(mockErrors));
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const mockErrors = generateMockErrors();
    setErrors(mockErrors);
    setMetrics(generateMetrics(mockErrors));
    setIsRefreshing(false);
  }, []);

  const toggleErrorExpansion = (errorId: string) => {
    const newExpanded = new Set(expandedErrors);
    if (newExpanded.has(errorId)) {
      newExpanded.delete(errorId);
    } else {
      newExpanded.add(errorId);
    }
    setExpandedErrors(newExpanded);
  };

  const markAsResolved = (errorId: string) => {
    setErrors((prev) =>
      prev.map((error) =>
        error.id === errorId ? { ...error, resolved: true } : error,
      ),
    );
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getErrorIcon = (level: ErrorLog["level"]) => {
    switch (level) {
      case "error":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-neon-yellow" />;
      case "info":
        return <Info className="h-4 w-4 text-neon-cyan" />;
      case "debug":
        return <Code className="h-4 w-4 text-gray-400" />;
    }
  };

  const getLevelColor = (level: ErrorLog["level"]) => {
    switch (level) {
      case "error":
        return "text-neon-red border-neon-red/30 bg-neon-red/10";
      case "warning":
        return "text-neon-yellow border-neon-yellow/30 bg-neon-yellow/10";
      case "info":
        return "text-neon-cyan border-neon-cyan/30 bg-neon-cyan/10";
      case "debug":
        return "text-gray-400 border-gray-400/30 bg-gray-400/10";
    }
  };

  const getCategoryIcon = (category: ErrorLog["category"]) => {
    switch (category) {
      case "system":
        return <Server className="h-4 w-4" />;
      case "api":
        return <Globe className="h-4 w-4" />;
      case "database":
        return <Database className="h-4 w-4" />;
      case "workflow":
        return <Zap className="h-4 w-4" />;
      case "security":
        return <AlertTriangle className="h-4 w-4" />;
      case "performance":
        return <Clock className="h-4 w-4" />;
    }
  };

  const filteredErrors = errors.filter((error) => {
    if (filter.level.length > 0 && !filter.level.includes(error.level))
      return false;
    if (filter.category.length > 0 && !filter.category.includes(error.category))
      return false;
    if (filter.source.length > 0 && !filter.source.includes(error.source))
      return false;
    if (filter.resolved !== null && error.resolved !== filter.resolved)
      return false;
    if (
      filter.searchTerm &&
      !error.message.toLowerCase().includes(filter.searchTerm.toLowerCase())
    )
      return false;
    return true;
  });

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-neon-cyan">Error Tracking</h2>
          <p className="text-neon-cyan/70">
            Monitor and manage system errors and warnings
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            onClick={() => setShowFilters(!showFilters)}
            variant="outline"
            size="sm"
            className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
          >
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>
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
          <Button
            onClick={() => {
              const csvContent = [
                "timestamp,level,message,source,category,resolved",
                ...filteredErrors.map(
                  (error) =>
                    `${error.timestamp},${error.level},${error.message},${error.source},${error.category},${error.resolved}`,
                ),
              ].join("\n");

              const blob = new Blob([csvContent], { type: "text/csv" });
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `error-log-${new Date().toISOString().split("T")[0]}.csv`;
              a.click();
              window.URL.revokeObjectURL(url);
            }}
            variant="outline"
            size="sm"
            className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <XCircle className="h-8 w-8 text-neon-red mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">
              {metrics.totalErrors}
            </div>
            <div className="text-sm text-neon-cyan/70">Total Errors</div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <AlertTriangle className="h-8 w-8 text-neon-yellow mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">
              {metrics.criticalErrors}
            </div>
            <div className="text-sm text-neon-cyan/70">Critical Errors</div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-white">
              {metrics.errorRate}%
            </div>
            <div className="text-sm text-neon-cyan/70">Error Rate</div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-white">
              {metrics.resolvedErrors}
            </div>
            <div className="text-sm text-neon-cyan/70">Resolved</div>
          </CardContent>
        </Card>

        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4 text-center">
            <Clock className="h-8 w-8 text-neon-cyan mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">
              {metrics.averageResolutionTime}h
            </div>
            <div className="text-sm text-neon-cyan/70">Avg Resolution</div>
          </CardContent>
        </Card>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <Card className="bg-cyber-dark border-neon-cyan/30">
          <CardContent className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium text-neon-cyan block mb-2">
                  Search
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neon-cyan/50 h-4 w-4" />
                  <Input
                    placeholder="Search error messages..."
                    value={filter.searchTerm}
                    onChange={(e) =>
                      setFilter((prev) => ({
                        ...prev,
                        searchTerm: e.target.value,
                      }))
                    }
                    className="pl-10 bg-cyber-dark border-neon-cyan/30 text-white"
                  />
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-neon-cyan block mb-2">
                  Error Level
                </label>
                <div className="flex flex-wrap gap-2">
                  {["error", "warning", "info", "debug"].map((level) => (
                    <Button
                      key={level}
                      onClick={() => {
                        setFilter((prev) => ({
                          ...prev,
                          level: prev.level.includes(level)
                            ? prev.level.filter((l) => l !== level)
                            : [...prev.level, level],
                        }));
                      }}
                      variant={
                        filter.level.includes(level) ? "default" : "outline"
                      }
                      size="sm"
                      className={cn(
                        "text-xs",
                        filter.level.includes(level)
                          ? "bg-neon-cyan text-black"
                          : "border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10",
                      )}
                    >
                      {level}
                    </Button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-neon-cyan block mb-2">
                  Status
                </label>
                <div className="flex gap-2">
                  <Button
                    onClick={() =>
                      setFilter((prev) => ({ ...prev, resolved: null }))
                    }
                    variant={filter.resolved === null ? "default" : "outline"}
                    size="sm"
                    className={cn(
                      "text-xs",
                      filter.resolved === null
                        ? "bg-neon-cyan text-black"
                        : "border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10",
                    )}
                  >
                    All
                  </Button>
                  <Button
                    onClick={() =>
                      setFilter((prev) => ({ ...prev, resolved: false }))
                    }
                    variant={filter.resolved === false ? "default" : "outline"}
                    size="sm"
                    className={cn(
                      "text-xs",
                      filter.resolved === false
                        ? "bg-neon-cyan text-black"
                        : "border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10",
                    )}
                  >
                    Unresolved
                  </Button>
                  <Button
                    onClick={() =>
                      setFilter((prev) => ({ ...prev, resolved: true }))
                    }
                    variant={filter.resolved === true ? "default" : "outline"}
                    size="sm"
                    className={cn(
                      "text-xs",
                      filter.resolved === true
                        ? "bg-neon-cyan text-black"
                        : "border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10",
                    )}
                  >
                    Resolved
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error List */}
      <Card className="bg-cyber-dark border-neon-cyan/30">
        <CardHeader>
          <CardTitle className="text-neon-cyan flex items-center justify-between">
            <span>Recent Errors ({filteredErrors.length})</span>
            <Badge className="text-neon-cyan border-neon-cyan/30 bg-neon-cyan/10">
              Live
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {filteredErrors.slice(0, maxErrors).map((error) => (
              <div
                key={error.id}
                className={cn(
                  "p-4 rounded-lg border transition-all duration-200",
                  error.resolved
                    ? "border-neon-green/20 bg-neon-green/5"
                    : "border-neon-cyan/20 bg-neon-cyan/5 hover:bg-neon-cyan/10",
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      {getErrorIcon(error.level)}
                      <Badge
                        className={getLevelColor(error.level)}
                        variant="outline"
                      >
                        {error.level.toUpperCase()}
                      </Badge>
                      <div className="flex items-center gap-1 text-neon-cyan/70">
                        {getCategoryIcon(error.category)}
                        <span className="text-xs">{error.category}</span>
                      </div>
                      <span className="text-xs text-neon-cyan/50">
                        {new Date(error.timestamp).toLocaleString()}
                      </span>
                      {error.resolved && (
                        <Badge className="text-neon-green border-neon-green/30 bg-neon-green/10">
                          Resolved
                        </Badge>
                      )}
                    </div>

                    <h4 className="font-medium text-white mb-1">
                      {error.message}
                    </h4>

                    <div className="flex items-center gap-4 text-xs text-neon-cyan/70 mb-2">
                      <span>Source: {error.source}</span>
                      {error.occurrences > 1 && (
                        <span>Occurrences: {error.occurrences}</span>
                      )}
                      {error.userId && <span>User: {error.userId}</span>}
                    </div>

                    {error.tags.length > 0 && (
                      <div className="flex gap-1 mb-2">
                        {error.tags.map((tag) => (
                          <Badge
                            key={tag}
                            variant="outline"
                            className="text-xs text-neon-magenta border-neon-magenta/30"
                          >
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      onClick={() => toggleErrorExpansion(error.id)}
                      variant="ghost"
                      size="sm"
                      className="text-neon-cyan hover:bg-neon-cyan/10"
                    >
                      {expandedErrors.has(error.id) ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )}
                    </Button>
                    {!error.resolved && (
                      <Button
                        onClick={() => markAsResolved(error.id)}
                        variant="outline"
                        size="sm"
                        className="text-neon-green border-neon-green/50 hover:bg-neon-green/10"
                      >
                        Resolve
                      </Button>
                    )}
                  </div>
                </div>

                {/* Expanded Details */}
                {expandedErrors.has(error.id) && (
                  <div className="mt-4 pt-4 border-t border-neon-cyan/20">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <div>
                        <h5 className="text-sm font-medium text-neon-cyan mb-2">
                          Context
                        </h5>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-neon-cyan/70">
                              Request ID:
                            </span>
                            <span className="text-white font-mono">
                              {error.requestId}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-neon-cyan/70">
                              Session ID:
                            </span>
                            <span className="text-white font-mono">
                              {error.sessionId}
                            </span>
                          </div>
                          {error.context &&
                            Object.entries(error.context).map(
                              ([key, value]) => (
                                <div key={key} className="flex justify-between">
                                  <span className="text-neon-cyan/70">
                                    {key}:
                                  </span>
                                  <span className="text-white font-mono">
                                    {String(value)}
                                  </span>
                                </div>
                              ),
                            )}
                        </div>
                      </div>

                      {error.stackTrace && (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <h5 className="text-sm font-medium text-neon-cyan">
                              Stack Trace
                            </h5>
                            <Button
                              onClick={() => copyToClipboard(error.stackTrace!)}
                              variant="ghost"
                              size="sm"
                              className="text-neon-cyan hover:bg-neon-cyan/10"
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                          </div>
                          <pre className="text-xs bg-black/50 p-3 rounded border border-neon-cyan/20 overflow-auto max-h-48 font-mono text-neon-cyan/70">
                            {error.stackTrace}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {filteredErrors.length === 0 && (
              <div className="text-center py-8 text-neon-cyan/50">
                No errors found matching the current filters.
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Top Error Sources */}
      <Card className="bg-cyber-dark border-neon-cyan/30">
        <CardHeader>
          <CardTitle className="text-neon-cyan">Top Error Sources</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {metrics.topErrorSources.map((source, index) => (
              <div
                key={source.source}
                className="flex items-center justify-between"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-neon-cyan/20 flex items-center justify-center text-sm font-bold text-neon-cyan">
                    {index + 1}
                  </div>
                  <span className="text-white">{source.source}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-neon-cyan font-medium">
                    {source.count}
                  </span>
                  <span className="text-neon-cyan/70 text-sm">errors</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ErrorTracking;
