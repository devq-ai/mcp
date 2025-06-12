"use client";

import * as React from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import { Clock, Settings, CheckCircle, XCircle, Timer } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function DelayNode({ data, selected }: NodeProps) {
  const nodeData = data as any;
  const {
    label,
    config,
    isExecuting,
    executionStatus,
    executionResult,
    remainingTime,
  } = nodeData;

  const getStatusColor = () => {
    switch (executionStatus) {
      case "running":
        return "border-neon-green bg-neon-green/10 animate-pulse";
      case "completed":
        return "border-neon-green bg-neon-green/10";
      case "error":
        return "border-neon-red bg-neon-red/10";
      default:
        return "border-neon-green bg-neon-green/5";
    }
  };

  const getStatusIcon = () => {
    switch (executionStatus) {
      case "running":
        return <Timer className="h-4 w-4 text-neon-green animate-pulse" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-neon-green" />;
      case "error":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      default:
        return <Clock className="h-4 w-4 text-neon-green" />;
    }
  };

  const getDurationInMs = () => {
    switch (config.unit) {
      case "seconds":
        return config.duration * 1000;
      case "minutes":
        return config.duration * 60 * 1000;
      case "hours":
        return config.duration * 60 * 60 * 1000;
      default:
        return config.duration;
    }
  };

  const formatDuration = () => {
    if (!config.duration) return "Configure Duration";
    return `${config.duration} ${config.unit}`;
  };

  const formatRemainingTime = () => {
    if (!remainingTime) return "";

    if (remainingTime < 1000) {
      return `${remainingTime}ms`;
    } else if (remainingTime < 60000) {
      return `${Math.ceil(remainingTime / 1000)}s`;
    } else if (remainingTime < 3600000) {
      return `${Math.ceil(remainingTime / 60000)}m`;
    } else {
      return `${Math.ceil(remainingTime / 3600000)}h`;
    }
  };

  const getProgressPercentage = () => {
    if (!isExecuting || !remainingTime) return 0;
    const totalDuration = getDurationInMs();
    const elapsed = totalDuration - remainingTime;
    return Math.min(100, Math.max(0, (elapsed / totalDuration) * 100));
  };

  return (
    <Card
      className={cn(
        "min-w-[200px] transition-all duration-200 backdrop-blur-sm",
        selected
          ? "ring-2 ring-neon-green shadow-neon-green/50 shadow-lg"
          : "hover:shadow-md",
        getStatusColor(),
      )}
    >
      <CardContent className="p-4">
        {/* Input Handle */}
        <Handle
          type="target"
          position={Position.Left}
          className="w-3 h-3 bg-neon-green border-2 border-cyber-dark hover:bg-neon-green/80"
          style={{ left: -6 }}
        />

        {/* Node Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className="font-semibold text-white text-sm">{label}</span>
          </div>
          <Badge
            variant="outline"
            className="text-xs text-neon-green border-neon-green/50"
          >
            DELAY
          </Badge>
        </div>

        {/* Node Content */}
        <div className="space-y-2">
          <div className="text-sm text-white/90">{formatDuration()}</div>

          {config.description && (
            <div className="text-xs text-neon-green/70">
              {config.description}
            </div>
          )}

          {config.duration && (
            <div className="flex items-center gap-1">
              <Settings className="h-3 w-3 text-neon-magenta" />
              <span className="text-xs text-neon-magenta">
                {getDurationInMs()}ms total
              </span>
            </div>
          )}
        </div>

        {/* Execution Status */}
        {isExecuting && remainingTime && (
          <div className="mt-3 pt-2 border-t border-neon-green/20">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-neon-green">Waiting...</span>
                <span className="text-xs text-neon-green/70">
                  {formatRemainingTime()}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-cyber-dark/50 rounded-full h-1.5">
                <div
                  className="bg-neon-green rounded-full h-1.5 transition-all duration-1000 ease-linear"
                  style={{ width: `${getProgressPercentage()}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Execution Result */}
        {executionResult && executionStatus === "completed" && (
          <div className="mt-3 pt-2 border-t border-neon-green/20">
            <div className="text-xs text-neon-green">
              ✓ Delay completed
              <div className="text-neon-cyan/70 mt-1">
                Duration: {executionResult.actualDuration}ms
              </div>
            </div>
          </div>
        )}

        {executionStatus === "error" && (
          <div className="mt-3 pt-2 border-t border-neon-red/20">
            <div className="text-xs text-neon-red">✗ Delay interrupted</div>
          </div>
        )}

        {/* Output Handle */}
        <Handle
          type="source"
          position={Position.Right}
          className="w-3 h-3 bg-neon-green border-2 border-cyber-dark hover:bg-neon-green/80"
          style={{ right: -6 }}
        />
      </CardContent>
    </Card>
  );
}
