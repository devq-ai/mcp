"use client";

import * as React from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import { GitBranch, Settings, CheckCircle, XCircle, Clock } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function ConditionNode({ data, selected }: NodeProps) {
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
        return "border-neon-yellow bg-neon-yellow/5";
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
        return <GitBranch className="h-4 w-4 text-neon-yellow" />;
    }
  };

  const getConditionDisplayName = () => {
    if (!config.condition) return "Configure Condition";
    return config.condition.length > 30
      ? config.condition.substring(0, 30) + "..."
      : config.condition;
  };

  const getOperatorSymbol = () => {
    switch (config.operator) {
      case "equals":
        return "==";
      case "not_equals":
        return "!=";
      case "greater_than":
        return ">";
      case "less_than":
        return "<";
      case "contains":
        return "∋";
      case "regex":
        return "~/";
      default:
        return "?";
    }
  };

  return (
    <Card
      className={cn(
        "min-w-[220px] transition-all duration-200 backdrop-blur-sm",
        selected
          ? "ring-2 ring-neon-yellow shadow-neon-yellow/50 shadow-lg"
          : "hover:shadow-md",
        getStatusColor(),
      )}
    >
      <CardContent className="p-4">
        {/* Input Handle */}
        <Handle
          type="target"
          position={Position.Left}
          className="w-3 h-3 bg-neon-yellow border-2 border-cyber-dark hover:bg-neon-yellow/80"
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
            className="text-xs text-neon-yellow border-neon-yellow/50"
          >
            CONDITION
          </Badge>
        </div>

        {/* Node Content */}
        <div className="space-y-2">
          <div className="text-sm text-white/90">
            {getConditionDisplayName()}
          </div>

          {config.operator && config.variable && (
            <div className="text-xs text-neon-yellow/70 font-mono">
              {config.variable} {getOperatorSymbol()} {config.value}
            </div>
          )}

          {(config.condition || config.operator) && (
            <div className="flex items-center gap-1">
              <Settings className="h-3 w-3 text-neon-magenta" />
              <span className="text-xs text-neon-magenta">Configured</span>
            </div>
          )}
        </div>

        {/* Branch Labels */}
        <div className="mt-3 flex justify-between text-xs">
          <div className="text-neon-green">{config.trueLabel || "True"}</div>
          <div className="text-neon-red">{config.falseLabel || "False"}</div>
        </div>

        {/* Execution Status */}
        {isExecuting && (
          <div className="mt-3 pt-2 border-t border-neon-yellow/20">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-yellow rounded-full animate-pulse" />
              <span className="text-xs text-neon-yellow">Evaluating...</span>
            </div>
          </div>
        )}

        {/* Execution Result */}
        {executionResult && executionStatus === "completed" && (
          <div className="mt-3 pt-2 border-t border-neon-green/20">
            <div className="text-xs">
              <span
                className={
                  executionResult.result ? "text-neon-green" : "text-neon-red"
                }
              >
                →{" "}
                {executionResult.result ? config.trueLabel : config.falseLabel}
              </span>
              {executionResult.value !== undefined && (
                <div className="text-neon-cyan/70 mt-1">
                  Value: {JSON.stringify(executionResult.value)}
                </div>
              )}
            </div>
          </div>
        )}

        {executionStatus === "error" && (
          <div className="mt-3 pt-2 border-t border-neon-red/20">
            <div className="text-xs text-neon-red">✗ Evaluation failed</div>
          </div>
        )}

        {/* Output Handles */}
        <Handle
          type="source"
          position={Position.Right}
          id="true"
          className="w-3 h-3 bg-neon-green border-2 border-cyber-dark hover:bg-neon-green/80"
          style={{ right: -6, top: "40%" }}
        />
        <Handle
          type="source"
          position={Position.Right}
          id="false"
          className="w-3 h-3 bg-neon-red border-2 border-cyber-dark hover:bg-neon-red/80"
          style={{ right: -6, top: "60%" }}
        />
      </CardContent>
    </Card>
  );
}
