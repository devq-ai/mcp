"use client";

import * as React from "react";
import Link from "next/link";
import {
  Zap,
  Bot,
  Settings,
  Network,
  Activity,
  CheckCircle,
  XCircle,
  Clock,
  Play,
  Pause,
  RotateCcw,
  Eye,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
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

export interface Agent {
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
  health_score: number;
}

interface AgentOverviewCardProps {
  agent: Agent;
  isSelected?: boolean;
  onSelect?: () => void;
  onAction?: (action: "start" | "stop" | "restart") => void;
  compact?: boolean;
}

export function AgentOverviewCard({
  agent,
  isSelected = false,
  onSelect,
  onAction,
  compact = false,
}: AgentOverviewCardProps) {
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

  const getPerformanceTrend = () => {
    // Mock trend calculation - in real app would compare with historical data
    const trend = Math.random() > 0.5 ? "up" : "down";
    const percentage = (Math.random() * 10).toFixed(1);

    return {
      direction: trend,
      percentage,
      icon:
        trend === "up" ? (
          <TrendingUp className="h-3 w-3 text-neon-green" />
        ) : (
          <TrendingDown className="h-3 w-3 text-neon-red" />
        ),
    };
  };

  const performanceTrend = getPerformanceTrend();

  const handleActionClick = (
    e: React.MouseEvent,
    action: "start" | "stop" | "restart",
  ) => {
    e.stopPropagation();
    onAction?.(action);
  };

  return (
    <Card
      className={cn(
        "cyber-card cursor-pointer transition-all hover:border-primary/50",
        isSelected && "border-primary shadow-neon",
        compact && "p-2",
      )}
      onClick={onSelect}
    >
      <CardHeader className={cn("pb-3", compact && "pb-2")}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {getAgentTypeIcon(agent.type)}
            <div className="min-w-0 flex-1">
              <CardTitle
                className={cn("text-lg truncate", compact && "text-base")}
              >
                {agent.name}
              </CardTitle>
              <CardDescription className="text-sm">
                v{agent.version} • {formatDuration(agent.uptime)}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center space-x-2 flex-shrink-0">
            <Badge className={getStatusColor(agent.status)}>
              {agent.status}
            </Badge>
            {getStatusIcon(agent.status)}
          </div>
        </div>
      </CardHeader>

      <CardContent className={cn("space-y-4", compact && "space-y-3")}>
        {/* Health Score */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Health Score</span>
          <div className="flex items-center space-x-2">
            <span
              className={cn(
                "text-lg font-bold",
                getHealthColor(agent.health_score),
              )}
            >
              {agent.health_score}%
            </span>
            {performanceTrend.icon}
          </div>
        </div>

        {!compact && (
          <>
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
              <div className="flex justify-between text-sm">
                <span>Executions</span>
                <span className="font-medium">
                  {agent.performance_metrics.total_executions.toLocaleString()}
                </span>
              </div>
            </div>

            {/* Resource Usage */}
            <div className="space-y-3">
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>CPU Usage</span>
                  <span>{agent.performance_metrics.cpu_usage.toFixed(1)}%</span>
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

            {/* Capabilities Preview */}
            <div>
              <div className="text-sm text-muted-foreground mb-2">
                Capabilities
              </div>
              <div className="flex flex-wrap gap-1">
                {agent.capabilities.slice(0, 3).map((capability) => (
                  <Badge key={capability} variant="outline" className="text-xs">
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

            {/* Error Indicator */}
            {agent.performance_metrics.error_count_24h > 0 && (
              <div className="flex items-center space-x-2 text-sm text-neon-orange">
                <AlertTriangle className="h-4 w-4" />
                <span>
                  {agent.performance_metrics.error_count_24h} errors in 24h
                </span>
              </div>
            )}

            {/* Last Heartbeat */}
            <div className="text-xs text-muted-foreground">
              Last heartbeat: {formatRelativeTime(agent.last_heartbeat)}
            </div>
          </>
        )}

        {/* Compact view metrics */}
        {compact && (
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div className="text-muted-foreground">Tasks</div>
              <div className="font-medium text-neon-lime">
                {agent.current_tasks}
              </div>
            </div>
            <div className="text-center">
              <div className="text-muted-foreground">Success</div>
              <div className="font-medium text-neon-green">
                {agent.performance_metrics.success_rate.toFixed(0)}%
              </div>
            </div>
            <div className="text-center">
              <div className="text-muted-foreground">CPU</div>
              <div className="font-medium">
                {agent.performance_metrics.cpu_usage.toFixed(0)}%
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        {isSelected && (
          <div className="flex space-x-2 pt-2 border-t border-border/50">
            <Link href={`/agents/${agent.id}`} className="flex-1">
              <Button size="sm" variant="outline" className="w-full">
                <Eye className="h-4 w-4 mr-2" />
                Details
              </Button>
            </Link>

            {agent.status === "offline" ? (
              <Button
                size="sm"
                onClick={(e) => handleActionClick(e, "start")}
                className="btn-neon text-neon-green"
              >
                <Play className="h-4 w-4 mr-1" />
                Start
              </Button>
            ) : (
              <div className="flex space-x-1">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={(e) => handleActionClick(e, "restart")}
                  title="Restart Agent"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={(e) => handleActionClick(e, "stop")}
                  className="text-neon-red border-neon-red hover:bg-neon-red/10"
                  title="Stop Agent"
                >
                  <Pause className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
