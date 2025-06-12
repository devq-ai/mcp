"use client";

import * as React from "react";
import { useState, useEffect, useCallback, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  ConnectionLineType,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Play,
  CheckCircle,
  AlertCircle,
  Clock,
  Zap,
  Database,
  GitBranch,
  Maximize2,
  Minimize2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Target,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface WorkflowStep {
  id: string;
  name: string;
  type: "start" | "action" | "condition" | "api" | "delay" | "end";
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  progress: number;
  duration?: number;
  agent_name?: string;
  position: { x: number; y: number };
  connections: string[];
  metadata?: Record<string, any>;
}

interface WorkflowExecutionData {
  id: string;
  name: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  currentStep?: string;
  executionPath: string[];
  steps: WorkflowStep[];
  startTime: string;
  endTime?: string;
}

interface WorkflowVisualizationProps {
  executionData: WorkflowExecutionData;
  onStepSelect?: (stepId: string) => void;
  onExecutionPathToggle?: (enabled: boolean) => void;
  className?: string;
  height?: number;
  readOnly?: boolean;
}

// Custom Node Component for Workflow Visualization
function WorkflowNode({ data, selected }: any) {
  const { step, isCurrentStep, isInExecutionPath, executionOrder } = data;

  const getStatusIcon = () => {
    switch (step.status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-neon-green" />;
      case "running":
        return <Play className="h-4 w-4 text-neon-lime animate-pulse" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-neon-red" />;
      case "pending":
        return <Clock className="h-4 w-4 text-neon-cyan/50" />;
      case "skipped":
        return <Clock className="h-4 w-4 text-gray-400" />;
      default:
        return <Clock className="h-4 w-4 text-neon-cyan/50" />;
    }
  };

  const getTypeIcon = () => {
    switch (step.type) {
      case "action":
        return <Zap className="h-3 w-3 text-neon-cyan" />;
      case "api":
        return <Database className="h-3 w-3 text-neon-magenta" />;
      case "condition":
        return <GitBranch className="h-3 w-3 text-neon-yellow" />;
      default:
        return null;
    }
  };

  const getNodeStyle = () => {
    let baseClasses = "min-w-[180px] transition-all duration-300";

    if (isCurrentStep) {
      baseClasses +=
        " ring-2 ring-neon-lime shadow-neon-lime/50 shadow-lg animate-pulse";
    } else if (selected) {
      baseClasses += " ring-2 ring-neon-cyan shadow-neon-cyan/50 shadow-lg";
    } else if (isInExecutionPath) {
      baseClasses +=
        " ring-1 ring-neon-magenta/50 shadow-neon-magenta/30 shadow-md";
    }

    switch (step.status) {
      case "completed":
        baseClasses += " border-neon-green bg-neon-green/10";
        break;
      case "running":
        baseClasses += " border-neon-lime bg-neon-lime/10";
        break;
      case "failed":
        baseClasses += " border-neon-red bg-neon-red/10";
        break;
      case "pending":
        baseClasses += " border-neon-cyan/30 bg-neon-cyan/5";
        break;
      case "skipped":
        baseClasses += " border-gray-500/30 bg-gray-500/5";
        break;
      default:
        baseClasses += " border-neon-cyan/30 bg-cyber-dark";
    }

    return baseClasses;
  };

  return (
    <Card className={cn("backdrop-blur-sm", getNodeStyle())}>
      <CardContent className="p-3">
        {/* Execution Order Badge */}
        {isInExecutionPath && executionOrder !== undefined && (
          <div className="absolute -top-2 -right-2 w-6 h-6 bg-neon-magenta rounded-full flex items-center justify-center text-xs font-bold text-black">
            {executionOrder}
          </div>
        )}

        {/* Node Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            {getTypeIcon()}
            <span className="font-semibold text-white text-sm truncate">
              {step.name}
            </span>
          </div>
          <Badge
            variant="outline"
            className="text-xs text-neon-cyan border-neon-cyan/50"
          >
            {step.type.toUpperCase()}
          </Badge>
        </div>

        {/* Progress Bar for Running Steps */}
        {step.status === "running" && (
          <div className="mb-2">
            <Progress value={step.progress} className="h-2" />
            <div className="text-xs text-neon-lime mt-1">
              {step.progress.toFixed(0)}% complete
            </div>
          </div>
        )}

        {/* Agent and Duration Info */}
        <div className="flex items-center justify-between text-xs">
          {step.agent_name && (
            <span className="text-neon-cyan/70">{step.agent_name}</span>
          )}
          {step.duration && (
            <span className="text-neon-cyan/70">
              {step.duration < 60
                ? `${step.duration}s`
                : `${Math.floor(step.duration / 60)}m ${step.duration % 60}s`}
            </span>
          )}
        </div>

        {/* Current Step Indicator */}
        {isCurrentStep && (
          <div className="mt-2 flex items-center gap-1 text-xs text-neon-lime">
            <div className="w-2 h-2 bg-neon-lime rounded-full animate-pulse" />
            <span>Executing now</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

const nodeTypes = {
  workflowNode: WorkflowNode,
};

export function WorkflowVisualization({
  executionData,
  onStepSelect,
  onExecutionPathToggle,
  className,
  height = 600,
  readOnly = true,
}: WorkflowVisualizationProps) {
  const initialNodes: Node[] = [];
  const initialEdges: Edge[] = [];
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  const [showExecutionPath, setShowExecutionPath] = useState(true);

  const [isFullscreen, setIsFullscreen] = useState(false);

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Convert workflow steps to React Flow nodes
  useEffect(() => {
    const flowNodes: Node[] = executionData.steps.map((step) => {
      const isCurrentStep = step.id === executionData.currentStep;
      const isInExecutionPath =
        showExecutionPath && executionData.executionPath.includes(step.id);
      const executionOrder = isInExecutionPath
        ? executionData.executionPath.indexOf(step.id) + 1
        : undefined;

      return {
        id: step.id,
        type: "workflowNode",
        position: step.position,
        data: {
          step,
          isCurrentStep,
          isInExecutionPath,
          executionOrder,
        },
        draggable: !readOnly,
        selectable: true,
      };
    });

    // Create edges based on step connections
    const flowEdges: Edge[] = [];
    executionData.steps.forEach((step) => {
      step.connections.forEach((targetId) => {
        const isInExecutionPath =
          showExecutionPath &&
          executionData.executionPath.includes(step.id) &&
          executionData.executionPath.includes(targetId);

        flowEdges.push({
          id: `${step.id}-${targetId}`,
          source: step.id,
          target: targetId,
          type: "smoothstep",
          animated: isInExecutionPath || step.status === "running",
          style: {
            stroke: isInExecutionPath
              ? "#FF0090" // neon-magenta for execution path
              : step.status === "completed"
                ? "#39FF14" // neon-green for completed
                : step.status === "running"
                  ? "#C7EA46" // neon-lime for running
                  : "#00FFFF", // neon-cyan for default
            strokeWidth: isInExecutionPath ? 3 : 2,
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: isInExecutionPath
              ? "#FF0090"
              : step.status === "completed"
                ? "#39FF14"
                : "#00FFFF",
          },
        });
      });
    });

    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [executionData, showExecutionPath, setNodes, setEdges, readOnly]);

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      setSelectedStep(node.id);
      onStepSelect?.(node.id);
    },
    [onStepSelect],
  );

  const handleFitView = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.fitView({ padding: 0.2 });
    }
  }, [reactFlowInstance]);

  const handleZoomIn = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.zoomIn();
    }
  }, [reactFlowInstance]);

  const handleZoomOut = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.zoomOut();
    }
  }, [reactFlowInstance]);

  const handleCenterOnCurrentStep = useCallback(() => {
    if (reactFlowInstance && executionData.currentStep) {
      const node = nodes.find((n) => n.id === executionData.currentStep);
      if (node) {
        reactFlowInstance.setCenter(node.position.x, node.position.y, {
          zoom: 1.2,
        });
      }
    }
  }, [reactFlowInstance, executionData.currentStep, nodes]);

  const toggleExecutionPath = useCallback(() => {
    const newValue = !showExecutionPath;
    setShowExecutionPath(newValue);
    onExecutionPathToggle?.(newValue);
  }, [showExecutionPath, onExecutionPathToggle]);

  const getExecutionStats = () => {
    const total = executionData.steps.length;
    const completed = executionData.steps.filter(
      (s) => s.status === "completed",
    ).length;
    const failed = executionData.steps.filter(
      (s) => s.status === "failed",
    ).length;
    const running = executionData.steps.filter(
      (s) => s.status === "running",
    ).length;

    return { total, completed, failed, running };
  };

  const stats = getExecutionStats();

  return (
    <div className={cn("relative", className)}>
      {/* Controls Panel */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
        <Card className="bg-cyber-dark/90 border-neon-cyan/30 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-neon-cyan">
              Workflow Controls
            </CardTitle>
          </CardHeader>
          <CardContent className="p-3 space-y-2">
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleFitView}
                className="text-neon-cyan hover:bg-neon-cyan/10"
              >
                <Target className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleZoomIn}
                className="text-neon-cyan hover:bg-neon-cyan/10"
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleZoomOut}
                className="text-neon-cyan hover:bg-neon-cyan/10"
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsFullscreen(!isFullscreen)}
                className="text-neon-cyan hover:bg-neon-cyan/10"
              >
                {isFullscreen ? (
                  <Minimize2 className="h-4 w-4" />
                ) : (
                  <Maximize2 className="h-4 w-4" />
                )}
              </Button>
            </div>

            {executionData.currentStep && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCenterOnCurrentStep}
                className="w-full text-neon-lime hover:bg-neon-lime/10"
              >
                <Play className="h-4 w-4 mr-2" />
                Focus Current
              </Button>
            )}

            <Button
              variant="ghost"
              size="sm"
              onClick={toggleExecutionPath}
              className={cn(
                "w-full",
                showExecutionPath
                  ? "text-neon-magenta hover:bg-neon-magenta/10"
                  : "text-neon-cyan/50 hover:bg-neon-cyan/10",
              )}
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Execution Path
            </Button>
          </CardContent>
        </Card>

        {/* Stats Panel */}
        <Card className="bg-cyber-dark/90 border-neon-cyan/30 backdrop-blur-sm">
          <CardContent className="p-3">
            <div className="text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-neon-cyan/70">Total:</span>
                <span className="text-white">{stats.total}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neon-green">Completed:</span>
                <span className="text-neon-green">{stats.completed}</span>
              </div>
              {stats.running > 0 && (
                <div className="flex justify-between">
                  <span className="text-neon-lime">Running:</span>
                  <span className="text-neon-lime">{stats.running}</span>
                </div>
              )}
              {stats.failed > 0 && (
                <div className="flex justify-between">
                  <span className="text-neon-red">Failed:</span>
                  <span className="text-neon-red">{stats.failed}</span>
                </div>
              )}
              <div className="pt-1 border-t border-neon-cyan/20">
                <div className="flex justify-between">
                  <span className="text-neon-cyan/70">Progress:</span>
                  <span className="text-white">
                    {((stats.completed / stats.total) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Workflow Visualization */}
      <div
        ref={reactFlowWrapper}
        className={cn(
          "bg-cyber-dark rounded-lg border border-neon-cyan/30",
          isFullscreen && "fixed inset-0 z-50 rounded-none",
        )}
        style={{ height: isFullscreen ? "100vh" : height }}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          onInit={setReactFlowInstance}
          nodeTypes={nodeTypes}
          connectionLineType={ConnectionLineType.SmoothStep}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          className="workflow-visualization"
          minZoom={0.1}
          maxZoom={4}
          defaultEdgeOptions={{
            animated: false,
            style: { strokeWidth: 2 },
          }}
        >
          <Background
            color="#00ffff"
            gap={20}
            size={1}
            className="opacity-10"
          />
          <Controls
            className="bg-cyber-dark/80 border border-neon-cyan/30"
            showZoom={false}
            showFitView={false}
            showInteractive={false}
          />
          <MiniMap
            className="bg-cyber-dark/80 border border-neon-cyan/30"
            nodeColor={(node) => {
              const step = node.data.step as WorkflowStep;
              switch (step.status) {
                case "completed":
                  return "#39FF14";
                case "running":
                  return "#C7EA46";
                case "failed":
                  return "#FF3131";
                default:
                  return "#00FFFF";
              }
            }}
            maskColor="rgba(0, 0, 0, 0.8)"
          />
        </ReactFlow>
      </div>

      {/* Selected Step Info */}
      {selectedStep && (
        <div className="absolute bottom-4 right-4 z-10">
          <Card className="bg-cyber-dark/90 border-neon-cyan/30 backdrop-blur-sm max-w-xs">
            <CardContent className="p-3">
              {(() => {
                const step = executionData.steps.find(
                  (s) => s.id === selectedStep,
                );
                if (!step) return null;

                return (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-white text-sm">
                        {step.name}
                      </h4>
                      <Badge variant="outline" className="text-xs">
                        {step.status}
                      </Badge>
                    </div>
                    {step.agent_name && (
                      <div className="text-xs text-neon-cyan/70">
                        Agent: {step.agent_name}
                      </div>
                    )}
                    {step.duration && (
                      <div className="text-xs text-neon-cyan/70">
                        Duration: {step.duration}s
                      </div>
                    )}
                    {step.metadata && Object.keys(step.metadata).length > 0 && (
                      <div className="text-xs text-neon-cyan/50">
                        {Object.entries(step.metadata).map(([key, value]) => (
                          <div key={key}>
                            {key}: {String(value)}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })()}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default WorkflowVisualization;
