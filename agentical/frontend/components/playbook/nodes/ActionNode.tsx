"use client";

import * as React from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import { Zap, Settings, CheckCircle, XCircle, Clock } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function ActionNode({ data, selected }: NodeProps) {
  const nodeData = data as any;
  const { label, config, isExecuting, executionStatus, executionResult } =
    nodeData;

  const getStatusColor = () => {
    switch (executionStatus) {
      case "running":
        return "border-neon-yellow bg-neon-yellow/10 animate-pulse";
      case "completed":
        return "border-neon-green bg-neon-green/10";
      case "error":
        return "border-neon-red bg-neon-red/10";
      default:
        return "border-neon-cyan bg-neon-cyan/5";
    }
  };

  const getStatusIcon = () => {
    switch (executionStatus) {
      case "running":
        return <Clock className="h-4 w-4 text-neon-yellow animate-spin" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-neon-green" />;
      case "error":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      default:
        return <Zap className="h-4 w-4 text-neon-cyan" />;
    }
  };

  const getActionDisplayName = () => {
    if (!config.action) return "Configure Action";
    return config.action
      .split("_")
      .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  return (
    <Card
      className={cn(
        "min-w-[200px] transition-all duration-200 backdrop-blur-sm",
        selected
          ? "ring-2 ring-neon-cyan shadow-neon-cyan/50 shadow-lg"
          : "hover:shadow-md",
        getStatusColor(),
      )}
    >
      <CardContent className="p-4">
        {/* Input Handle */}
        <Handle
          type="target"
          position={Position.Left}
          className="w-3 h-3 bg-neon-cyan border-2 border-cyber-dark hover:bg-neon-cyan/80"
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
            className="text-xs text-neon-cyan border-neon-cyan/50"
          >
            ACTION
          </Badge>
        </div>

        {/* Node Content */}
        <div className="space-y-2">
          <div className="text-sm text-white/90">{getActionDisplayName()}</div>

          {config.action && (
            <div className="text-xs text-neon-cyan/70 font-mono">
              {config.action}
            </div>
          )}

          {Object.keys(config.parameters).length > 0 && (
            <div className="flex items-center gap-1">
              <Settings className="h-3 w-3 text-neon-magenta" />
              <span className="text-xs text-neon-magenta">
                {Object.keys(config.parameters).length} parameter(s)
              </span>
            </div>
          )}

          {config.timeout && (
            <div className="text-xs text-neon-yellow/70">
              Timeout: {config.timeout}ms
            </div>
          )}
        </div>

        {/* Execution Status */}
        {isExecuting && (
          <div className="mt-3 pt-2 border-t border-neon-cyan/20">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-yellow rounded-full animate-pulse" />
              <span className="text-xs text-neon-yellow">Executing...</span>
            </div>
          </div>
        )}

        {/* Execution Result */}
        {executionResult && executionStatus === "completed" && (
          <div className="mt-3 pt-2 border-t border-neon-green/20">
            <div className="text-xs text-neon-green">
              ✓ Completed successfully
            </div>
          </div>
        )}

        {executionStatus === "error" && (
          <div className="mt-3 pt-2 border-t border-neon-red/20">
            <div className="text-xs text-neon-red">✗ Execution failed</div>
          </div>
        )}

        {/* Output Handle */}
        <Handle
          type="source"
          position={Position.Right}
          className="w-3 h-3 bg-neon-cyan border-2 border-cyber-dark hover:bg-neon-cyan/80"
          style={{ right: -6 }}
        />
      </CardContent>
    </Card>
  );
}
