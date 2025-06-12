"use client";

import * as React from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import { Play, Settings } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function StartNode({ data, selected }: NodeProps) {
  const nodeData = data as any;
  const { label, config, isExecuting, executionStatus } = nodeData;

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
        return <Play className="h-4 w-4 text-neon-yellow animate-spin" />;
      case "completed":
        return <Play className="h-4 w-4 text-neon-green" />;
      case "error":
        return <Play className="h-4 w-4 text-neon-red" />;
      default:
        return <Play className="h-4 w-4 text-neon-cyan" />;
    }
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
            START
          </Badge>
        </div>

        {/* Node Content */}
        <div className="space-y-2">
          <div className="text-sm text-white/90">{config.name}</div>
          {config.description && (
            <div className="text-xs text-neon-cyan/70">
              {config.description}
            </div>
          )}
          {config.trigger && (
            <div className="flex items-center gap-1">
              <Settings className="h-3 w-3 text-neon-magenta" />
              <span className="text-xs text-neon-magenta">
                {config.trigger}
              </span>
            </div>
          )}
        </div>

        {/* Execution Status */}
        {isExecuting && (
          <div className="mt-3 pt-2 border-t border-neon-cyan/20">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-yellow rounded-full animate-pulse" />
              <span className="text-xs text-neon-yellow">Initializing...</span>
            </div>
          </div>
        )}
      </CardContent>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-neon-cyan border-2 border-cyber-dark hover:bg-neon-cyan/80"
        style={{ right: -6 }}
      />
    </Card>
  );
}
