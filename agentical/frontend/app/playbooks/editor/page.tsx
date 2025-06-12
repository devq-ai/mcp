"use client";

import * as React from "react";
import { useState, useCallback, useRef, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  ReactFlowProvider,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Play,
  Save,
  Download,
  Upload,
  Undo,
  Redo,
  Trash2,
  Eye,
  EyeOff,
  Zap,
  Database,
  GitBranch,
  Clock,
  CheckCircle,
  ArrowLeft,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

// Custom Node Types
import StartNode from "@/components/playbook/nodes/StartNode";
import ActionNode from "@/components/playbook/nodes/ActionNode";
import ConditionNode from "@/components/playbook/nodes/ConditionNode";
import EndNode from "@/components/playbook/nodes/EndNode";
import ApiNode from "@/components/playbook/nodes/ApiNode";
import DelayNode from "@/components/playbook/nodes/DelayNode";

// Node type definitions
const nodeTypes = {
  start: StartNode as any,
  action: ActionNode as any,
  condition: ConditionNode as any,
  end: EndNode as any,
  api: ApiNode as any,
  delay: DelayNode as any,
};

// Initial nodes and edges
const initialNodes: Node[] = [
  {
    id: "1",
    type: "start",
    position: { x: 100, y: 100 },
    data: {
      label: "Start",
      config: {
        name: "Playbook Start",
        description: "Entry point of the playbook",
      },
    },
  },
];

const initialEdges: Edge[] = [];

// Node templates for the palette
const nodeTemplates = [
  {
    type: "action",
    label: "Action",
    icon: Zap,
    description: "Execute an action",
    color: "text-neon-cyan",
  },
  {
    type: "condition",
    label: "Condition",
    icon: GitBranch,
    description: "Conditional logic",
    color: "text-neon-yellow",
  },
  {
    type: "api",
    label: "API Call",
    icon: Database,
    description: "Make API request",
    color: "text-neon-magenta",
  },
  {
    type: "delay",
    label: "Delay",
    icon: Clock,
    description: "Wait/pause execution",
    color: "text-neon-green",
  },
  {
    type: "end",
    label: "End",
    icon: CheckCircle,
    description: "End the playbook",
    color: "text-neon-red",
  },
];

interface PlaybookMetadata {
  id?: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  version: string;
  author: string;
}

function PlaybookEditorContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const playbookId = searchParams.get("id");
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // UI state
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showNodePanel, setShowNodePanel] = useState(true);
  const [showPropertyPanel, setShowPropertyPanel] = useState(true);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState<any>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Playbook metadata
  const [metadata, setMetadata] = useState<PlaybookMetadata>({
    name: "New Playbook",
    description: "A new automation playbook",
    category: "automation",
    tags: [],
    version: "1.0.0",
    author: "User",
  });

  // History for undo/redo
  const [history, setHistory] = useState<{ nodes: Node[]; edges: Edge[] }[]>(
    [],
  );
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Load playbook if editing existing one
  useEffect(() => {
    if (playbookId) {
      loadPlaybook(playbookId);
    }
  }, [playbookId]);

  const loadPlaybook = async (id: string) => {
    try {
      // TODO: Implement API call to load playbook
      console.log("Loading playbook:", id);
    } catch (error) {
      console.error("Failed to load playbook:", error);
    }
  };

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setIsDragOver(false);

      if (!reactFlowWrapper.current || !reactFlowInstance) return;

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData("application/reactflow");

      if (!type) return;

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: {
          label: type.charAt(0).toUpperCase() + type.slice(1),
          config: getDefaultNodeConfig(type),
        },
      };

      setNodes((nds) => nds.concat(newNode));
      saveToHistory();
    },
    [reactFlowInstance, setNodes],
  );

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const getDefaultNodeConfig = (type: string) => {
    switch (type) {
      case "action":
        return { action: "", parameters: {} };
      case "condition":
        return { condition: "", trueLabel: "Yes", falseLabel: "No" };
      case "api":
        return { method: "GET", url: "", headers: {}, body: {} };
      case "delay":
        return { duration: 1000, unit: "ms" };
      case "end":
        return { status: "success", message: "Playbook completed" };
      default:
        return {};
    }
  };

  const saveToHistory = () => {
    const newHistoryItem = { nodes, edges };
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newHistoryItem);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const undo = () => {
    if (historyIndex > 0) {
      const prevState = history[historyIndex - 1];
      if (prevState) {
        setNodes(prevState.nodes);
        setEdges(prevState.edges);
        setHistoryIndex(historyIndex - 1);
      }
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1];
      if (nextState) {
        setNodes(nextState.nodes);
        setEdges(nextState.edges);
        setHistoryIndex(historyIndex + 1);
      }
    }
  };

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const onNodeDelete = useCallback(() => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
      setEdges((eds) =>
        eds.filter(
          (e) => e.source !== selectedNode.id && e.target !== selectedNode.id,
        ),
      );
      setSelectedNode(null);
      saveToHistory();
    }
  }, [selectedNode, setNodes, setEdges]);

  const updateNodeData = (nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node,
      ),
    );
    saveToHistory();
  };

  const savePlaybook = async () => {
    try {
      const playbookData = {
        ...metadata,
        nodes,
        edges,
        updatedAt: new Date().toISOString(),
      };

      console.log("Saving playbook:", playbookData);
      // TODO: Implement API call to save playbook
    } catch (error) {
      console.error("Failed to save playbook:", error);
    }
  };

  const executePlaybook = async () => {
    setIsExecuting(true);
    try {
      const playbookData = { nodes, edges };
      console.log("Executing playbook:", playbookData);
      // TODO: Implement API call to execute playbook

      // Simulate execution progress
      setExecutionProgress({ currentNodeId: nodes[0]?.id, status: "running" });

      setTimeout(() => {
        setExecutionProgress({ status: "completed" });
        setIsExecuting(false);
      }, 3000);
    } catch (error) {
      console.error("Failed to execute playbook:", error);
      setIsExecuting(false);
    }
  };

  const exportPlaybook = () => {
    const playbookData = { ...metadata, nodes, edges };
    const dataStr = JSON.stringify(playbookData, null, 2);
    const dataUri =
      "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);

    const exportFileDefaultName = `${metadata.name.replace(/\s+/g, "_")}.json`;

    const linkElement = document.createElement("a");
    linkElement.setAttribute("href", dataUri);
    linkElement.setAttribute("download", exportFileDefaultName);
    linkElement.click();
  };

  const importPlaybook = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const playbookData = JSON.parse(e.target?.result as string);
          setMetadata(playbookData);
          setNodes(playbookData.nodes || []);
          setEdges(playbookData.edges || []);
          saveToHistory();
        } catch (error) {
          console.error("Failed to import playbook:", error);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="h-screen bg-cyber-dark flex flex-col">
      {/* Top Toolbar */}
      <div className="border-b border-neon-cyan/30 bg-black/50 backdrop-blur-sm">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.back()}
              className="text-neon-cyan hover:bg-neon-cyan/10"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Separator orientation="vertical" className="h-6" />
            <div>
              <h1 className="text-xl font-bold text-neon-cyan">
                {metadata.name}
              </h1>
              <p className="text-sm text-neon-cyan/60">
                {metadata.description}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={undo}
              disabled={historyIndex <= 0}
              className="text-neon-cyan hover:bg-neon-cyan/10"
            >
              <Undo className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={redo}
              disabled={historyIndex >= history.length - 1}
              className="text-neon-cyan hover:bg-neon-cyan/10"
            >
              <Redo className="h-4 w-4" />
            </Button>
            <Separator orientation="vertical" className="h-6" />

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowNodePanel(!showNodePanel)}
              className="text-neon-cyan hover:bg-neon-cyan/10"
            >
              {showNodePanel ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
              Nodes
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowPropertyPanel(!showPropertyPanel)}
              className="text-neon-cyan hover:bg-neon-cyan/10"
            >
              {showPropertyPanel ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
              Properties
            </Button>

            <Separator orientation="vertical" className="h-6" />

            <input
              type="file"
              accept=".json"
              onChange={importPlaybook}
              className="hidden"
              id="import-file"
            />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => document.getElementById("import-file")?.click()}
              className="text-neon-cyan hover:bg-neon-cyan/10"
            >
              <Upload className="h-4 w-4" />
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={exportPlaybook}
              className="text-neon-cyan hover:bg-neon-cyan/10"
            >
              <Download className="h-4 w-4" />
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={savePlaybook}
              className="text-neon-green hover:bg-neon-green/10"
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </Button>

            <Button
              size="sm"
              onClick={executePlaybook}
              disabled={isExecuting}
              className="btn-neon text-neon-magenta"
            >
              <Play className="h-4 w-4 mr-2" />
              {isExecuting ? "Running..." : "Run"}
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Node Palette */}
        {showNodePanel && (
          <div className="w-64 border-r border-neon-cyan/30 bg-black/30 backdrop-blur-sm">
            <div className="p-4">
              <h3 className="text-sm font-semibold text-neon-cyan mb-4">
                Node Types
              </h3>
              <div className="space-y-2">
                {nodeTemplates.map((template) => {
                  const Icon = template.icon;
                  return (
                    <div
                      key={template.type}
                      className="flex items-center p-3 rounded-lg border border-neon-cyan/20 bg-cyber-dark hover:bg-neon-cyan/5 cursor-move transition-colors"
                      draggable
                      onDragStart={(event) => onDragStart(event, template.type)}
                    >
                      <Icon className={cn("h-5 w-5 mr-3", template.color)} />
                      <div>
                        <div className="text-sm font-medium text-white">
                          {template.label}
                        </div>
                        <div className="text-xs text-neon-cyan/60">
                          {template.description}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Main Flow Area */}
        <div className="flex-1 relative">
          <ReactFlowProvider>
            <div
              ref={reactFlowWrapper}
              className={cn("w-full h-full", isDragOver && "bg-neon-cyan/5")}
            >
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onInit={setReactFlowInstance}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onNodeClick={onNodeClick}
                nodeTypes={nodeTypes}
                className="bg-cyber-dark"
                fitView
              >
                <Background
                  color="#00ffff"
                  gap={20}
                  size={1}
                  className="opacity-20"
                />
                <Controls className="bg-black/50 border border-neon-cyan/30" />
                <MiniMap
                  className="bg-black/80 border border-neon-cyan/30"
                  nodeColor="#00ffff"
                  maskColor="rgba(0, 0, 0, 0.8)"
                />

                {/* Execution Progress Overlay */}
                {executionProgress && (
                  <Panel position="top-center">
                    <Card className="bg-black/90 border-neon-magenta/30">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 text-neon-magenta">
                          <Play className="h-4 w-4 animate-pulse" />
                          <span className="text-sm font-medium">
                            {executionProgress.status === "running"
                              ? "Executing..."
                              : "Completed"}
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  </Panel>
                )}
              </ReactFlow>
            </div>
          </ReactFlowProvider>
        </div>

        {/* Properties Panel */}
        {showPropertyPanel && (
          <div className="w-80 border-l border-neon-cyan/30 bg-black/30 backdrop-blur-sm">
            <ScrollArea className="h-full">
              <div className="p-4">
                {selectedNode ? (
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-sm font-semibold text-neon-cyan">
                        Node Properties
                      </h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={onNodeDelete}
                        className="text-neon-red hover:bg-neon-red/10"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <Label className="text-neon-cyan">Node ID</Label>
                        <Input
                          value={selectedNode.id}
                          disabled
                          className="bg-cyber-dark border-neon-cyan/30 text-white"
                        />
                      </div>

                      <div>
                        <Label className="text-neon-cyan">Type</Label>
                        <Input
                          value={selectedNode.type || "unknown"}
                          disabled
                          className="bg-cyber-dark border-neon-cyan/30 text-white"
                        />
                      </div>

                      <div>
                        <Label className="text-neon-cyan">Label</Label>
                        <Input
                          value={String(selectedNode.data?.label || "")}
                          onChange={(e) =>
                            updateNodeData(selectedNode.id, {
                              label: e.target.value,
                            })
                          }
                          className="bg-cyber-dark border-neon-cyan/30 text-white"
                        />
                      </div>

                      {/* Node-specific configuration would go here */}
                      <div className="pt-4 border-t border-neon-cyan/20">
                        <h4 className="text-sm font-medium text-neon-cyan mb-2">
                          Configuration
                        </h4>
                        <div className="text-xs text-neon-cyan/60">
                          Node-specific settings will be displayed here based on
                          node type.
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div>
                    <h3 className="text-sm font-semibold text-neon-cyan mb-4">
                      Playbook Settings
                    </h3>
                    <div className="space-y-4">
                      <div>
                        <Label className="text-neon-cyan">Name</Label>
                        <Input
                          value={metadata.name}
                          onChange={(e) =>
                            setMetadata({ ...metadata, name: e.target.value })
                          }
                          className="bg-cyber-dark border-neon-cyan/30 text-white"
                        />
                      </div>

                      <div>
                        <Label className="text-neon-cyan">Description</Label>
                        <Input
                          value={metadata.description}
                          onChange={(e) =>
                            setMetadata({
                              ...metadata,
                              description: e.target.value,
                            })
                          }
                          className="bg-cyber-dark border-neon-cyan/30 text-white"
                        />
                      </div>

                      <div>
                        <Label className="text-neon-cyan">Category</Label>
                        <Input
                          value={metadata.category}
                          onChange={(e) =>
                            setMetadata({
                              ...metadata,
                              category: e.target.value,
                            })
                          }
                          className="bg-cyber-dark border-neon-cyan/30 text-white"
                        />
                      </div>

                      <div>
                        <Label className="text-neon-cyan">Version</Label>
                        <Input
                          value={metadata.version}
                          onChange={(e) =>
                            setMetadata({
                              ...metadata,
                              version: e.target.value,
                            })
                          }
                          className="bg-cyber-dark border-neon-cyan/30 text-white"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
        )}
      </div>
    </div>
  );
}

export default function PlaybookEditorPage() {
  return (
    <Suspense
      fallback={
        <div className="h-screen bg-cyber-dark flex items-center justify-center">
          <div className="text-neon-cyan">Loading editor...</div>
        </div>
      }
    >
      <PlaybookEditorContent />
    </Suspense>
  );
}
