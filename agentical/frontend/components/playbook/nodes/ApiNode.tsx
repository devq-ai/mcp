"use client";

import * as React from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import {
  Database,
  Settings,
  CheckCircle,
  XCircle,
  Clock,
  Globe,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function ApiNode({ data, selected }: NodeProps) {
  const nodeData = data as any;
  const { label, config, isExecuting, executionStatus, executionResult } =
    nodeData;

  const getStatusColor = () => {
    switch (executionStatus) {
      case "running":
        return "border-neon-magenta bg-neon-magenta/10 animate-pulse";
      case "completed":
        return "border-neon-green bg-neon-green/10";
      case "error":
        return "border-neon-red bg-neon-red/10";
      default:
        return "border-neon-magenta bg-neon-magenta/5";
    }
  };

  const getStatusIcon = () => {
    switch (executionStatus) {
      case "running":
        return <Clock className="h-4 w-4 text-neon-magenta animate-spin" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-neon-green" />;
      case "error":
        return <XCircle className="h-4 w-4 text-neon-red" />;
      default:
        return <Database className="h-4 w-4 text-neon-magenta" />;
    }
  };

  const getMethodColor = () => {
    switch (config.method) {
      case "GET":
        return "text-neon-green border-neon-green/50";
      case "POST":
        return "text-neon-cyan border-neon-cyan/50";
      case "PUT":
        return "text-neon-yellow border-neon-yellow/50";
      case "DELETE":
        return "text-neon-red border-neon-red/50";
      case "PATCH":
        return "text-neon-magenta border-neon-magenta/50";
      default:
        return "text-neon-cyan border-neon-cyan/50";
    }
  };

  const getUrlDisplayName = () => {
    if (!config.url) return "Configure API Endpoint";
    try {
      const url = new URL(config.url);
      return `${url.hostname}${url.pathname}`;
    } catch {
      return config.url.length > 25
        ? config.url.substring(0, 25) + "..."
        : config.url;
    }
  };

  const getAuthDisplayName = () => {
    if (!config.authentication || config.authentication.type === "none") {
      return "No Auth";
    }
    return config.authentication.type.toUpperCase();
  };

  const hasConfiguration = () => {
    return config.url && config.method;
  };

  const getStatusText = () => {
    if (executionResult) {
      return `${executionResult.status} ${executionResult.statusText}`;
    }
    return "";
  };

  const getResponseSizeText = () => {
    if (executionResult?.data) {
      const size = JSON.stringify(executionResult.data).length;
      if (size > 1024) {
        return `${(size / 1024).toFixed(1)}KB`;
      }
      return `${size}B`;
    }
    return "";
  };

  return (
    <Card
      className={cn(
        "min-w-[240px] transition-all duration-200 backdrop-blur-sm",
        selected
          ? "ring-2 ring-neon-magenta shadow-neon-magenta/50 shadow-lg"
          : "hover:shadow-md",
        getStatusColor(),
      )}
    >
      <CardContent className="p-4">
        {/* Input Handle */}
        <Handle
          type="target"
          position={Position.Left}
          className="w-3 h-3 bg-neon-magenta border-2 border-cyber-dark hover:bg-neon-magenta/80"
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
            className={cn("text-xs border-neon-magenta/50", getMethodColor())}
          >
            {config.method || "API"}
          </Badge>
        </div>

        {/* Node Content */}
        <div className="space-y-2">
          <div className="text-sm text-white/90">{getUrlDisplayName()}</div>

          {config.url && (
            <div className="flex items-center gap-1">
              <Globe className="h-3 w-3 text-neon-cyan" />
              <span className="text-xs text-neon-cyan/70 font-mono">
                {config.method}
              </span>
            </div>
          )}

          {hasConfiguration() && (
            <div className="flex items-center gap-1">
              <Settings className="h-3 w-3 text-neon-magenta" />
              <span className="text-xs text-neon-magenta">
                {getAuthDisplayName()}
                {Object.keys(config.headers).length > 0 &&
                  ` • ${Object.keys(config.headers).length} header(s)`}
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
          <div className="mt-3 pt-2 border-t border-neon-magenta/20">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-magenta rounded-full animate-pulse" />
              <span className="text-xs text-neon-magenta">
                Making request...
              </span>
            </div>
          </div>
        )}

        {/* Execution Result */}
        {executionResult && executionStatus === "completed" && (
          <div className="mt-3 pt-2 border-t border-neon-green/20">
            <div className="text-xs space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-neon-green">✓ {getStatusText()}</span>
                <span className="text-neon-cyan/70">
                  {executionResult.duration}ms
                </span>
              </div>
              {getResponseSizeText() && (
                <div className="text-neon-cyan/70">
                  Response: {getResponseSizeText()}
                </div>
              )}
            </div>
          </div>
        )}

        {executionStatus === "error" && (
          <div className="mt-3 pt-2 border-t border-neon-red/20">
            <div className="text-xs text-neon-red">
              ✗ Request failed
              {executionResult && (
                <div className="mt-1">{getStatusText() || "Network error"}</div>
              )}
            </div>
          </div>
        )}

        {/* Output Handle */}
        <Handle
          type="source"
          position={Position.Right}
          className="w-3 h-3 bg-neon-magenta border-2 border-cyber-dark hover:bg-neon-magenta/80"
          style={{ right: -6 }}
        />
      </CardContent>
    </Card>
  );
}
