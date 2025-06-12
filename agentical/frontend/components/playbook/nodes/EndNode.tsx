"use client";

import * as React from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import {
  CheckCircle,
  XCircle,
  Settings,
  AlertCircle,
  Flag,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function EndNode({ data, selected }: NodeProps) {
  const nodeData = data as any;
  const { label, config, isExecuting, executionStatus, executionResult } =
    nodeData;

  const getStatusColor = () => {
    if (executionStatus === "completed") {
      switch (config.status) {
        case "success":
          return "border-neon-green bg-neon-green/10";
        case "error":
          return "border-neon-red bg-neon-red/10";
        case "cancelled":
          return "border-neon-yellow bg-neon-yellow/10";
        default:
          return "border-neon-cyan bg-neon-cyan/10";
      }
    }

    switch (executionStatus) {
      case "running":
        return "border-neon-cyan bg-neon-cyan/10 animate-pulse";
      case "error":
        return "border-neon-red bg-neon-red/10";
      default:
        return "border-neon-red bg-neon-red/5";
    }
  };

  const getStatusIcon = () => {
    if (executionStatus === "completed") {
      switch (config.status) {
        case "success":
          return <CheckCircle className="h-4 w-4 text-neon-green" />;
        case "error":
          return <XCircle className="h-4 w-4 text-neon-red" />;
        case "cancelled":
          return <AlertCircle className="h-4 w-4 text-neon-yellow" />;
        default:
          return <Flag className="h-4 w-4 text-neon-cyan" />;
      }
    }

    switch (executionStatus) {
      case "running":
        return <Flag className="h-4 w-4 text-neon-cyan animate-pulse" />;
      case "error":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      default:
        return <Flag className="h-4 w-4 text-neon-red" />;
    }
  };

  const getBadgeColor = () => {
    switch (config.status) {
      case "success":
        return "text-neon-green border-neon-green/50";
      case "error":
        return "text-neon-red border-neon-red/50";
      case "cancelled":
        return "text-neon-yellow border-neon-yellow/50";
      default:
        return "text-neon-red border-neon-red/50";
    }
  };

  const getMessageDisplayName = () => {
    if (!config.message) return "Configure End Message";
    return config.message.length > 40
      ? config.message.substring(0, 40) + "..."
      : config.message;
  };

  const hasNotifications = () => {
    return (
      config.notifications &&
      (config.notifications.email ||
        config.notifications.webhook ||
        config.notifications.slack)
    );
  };

  const getNotificationCount = () => {
    if (!config.notifications) return 0;
    let count = 0;
    if (config.notifications.email) count++;
    if (config.notifications.webhook) count++;
    if (config.notifications.slack) count++;
    return count;
  };

  return (
    <Card
      className={cn(
        "min-w-[200px] transition-all duration-200 backdrop-blur-sm",
        selected
          ? "ring-2 ring-neon-red shadow-neon-red/50 shadow-lg"
          : "hover:shadow-md",
        getStatusColor(),
      )}
    >
      <CardContent className="p-4">
        {/* Input Handle */}
        <Handle
          type="target"
          position={Position.Left}
          className="w-3 h-3 bg-neon-red border-2 border-cyber-dark hover:bg-neon-red/80"
          style={{ left: -6 }}
        />

        {/* Node Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className="font-semibold text-white text-sm">{label}</span>
          </div>
          <Badge variant="outline" className={cn("text-xs", getBadgeColor())}>
            END
          </Badge>
        </div>

        {/* Node Content */}
        <div className="space-y-2">
          <div className="text-sm text-white/90">{getMessageDisplayName()}</div>

          {config.status && (
            <div className="text-xs font-medium">
              <span
                className={cn(
                  config.status === "success"
                    ? "text-neon-green"
                    : config.status === "error"
                      ? "text-neon-red"
                      : config.status === "cancelled"
                        ? "text-neon-yellow"
                        : "text-neon-cyan",
                )}
              >
                {config.status.toUpperCase()}
              </span>
            </div>
          )}

          {(hasNotifications() || config.cleanup || config.outputData) && (
            <div className="flex items-center gap-1">
              <Settings className="h-3 w-3 text-neon-magenta" />
              <span className="text-xs text-neon-magenta">
                {hasNotifications() &&
                  `${getNotificationCount()} notification(s)`}
                {config.cleanup &&
                  (hasNotifications() ? " • " : "") + "Cleanup"}
                {config.outputData &&
                  (hasNotifications() || config.cleanup ? " • " : "") +
                    "Output"}
              </span>
            </div>
          )}
        </div>

        {/* Execution Status */}
        {isExecuting && (
          <div className="mt-3 pt-2 border-t border-neon-red/20">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-cyan rounded-full animate-pulse" />
              <span className="text-xs text-neon-cyan">Finalizing...</span>
            </div>
          </div>
        )}

        {/* Execution Result */}
        {executionResult && executionStatus === "completed" && (
          <div className="mt-3 pt-2 border-t border-neon-green/20">
            <div className="text-xs space-y-1">
              <div className="text-neon-green">✓ Playbook completed</div>
              <div className="text-neon-cyan/70">
                Status: {executionResult.finalStatus}
              </div>
              <div className="text-neon-cyan/70">
                Total time: {executionResult.executionTime}ms
              </div>
              {executionResult.outputData && (
                <div className="text-neon-cyan/70">
                  Output: {JSON.stringify(executionResult.outputData).length}B
                </div>
              )}
            </div>
          </div>
        )}

        {executionStatus === "error" && (
          <div className="mt-3 pt-2 border-t border-neon-red/20">
            <div className="text-xs text-neon-red">✗ Playbook failed</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
