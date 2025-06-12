"use client";

import * as React from "react";
import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import {
  Plus,
  Play,
  Edit,
  Clock,
  User,
  Search,
  Grid,
  List,
  ChevronRight,
  Zap,
  Bot,
  Send,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

interface Playbook {
  id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  author: string;
  createdAt: string;
  updatedAt: string;
  status: "draft" | "published" | "archived";
  executionCount: number;
  lastRun?: string;
}

interface ChatMessage {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: string;
}

function PlaybooksContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const action = searchParams.get("action");

  // State
  const [playbooks, setPlaybooks] = useState<Playbook[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [loading, setLoading] = useState(true);

  // Chat state for Super Agent interaction
  const [showChat, setShowChat] = useState(action === "chat");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      type: "assistant",
      content:
        "Hello! I'm the Super Agent. I can help you create, modify, and execute playbooks. What would you like to accomplish today?",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  // Mock data
  useEffect(() => {
    const mockPlaybooks: Playbook[] = [
      {
        id: "1",
        name: "Data Backup Automation",
        description:
          "Automated daily backup of critical databases with validation and notification",
        category: "automation",
        tags: ["backup", "database", "scheduling"],
        author: "System Admin",
        createdAt: "2024-01-15T10:00:00Z",
        updatedAt: "2024-01-20T14:30:00Z",
        status: "published",
        executionCount: 45,
        lastRun: "2024-01-27T02:00:00Z",
      },
      {
        id: "2",
        name: "User Onboarding Flow",
        description:
          "Complete user registration, profile setup, and welcome email sequence",
        category: "workflow",
        tags: ["onboarding", "users", "email"],
        author: "HR Team",
        createdAt: "2024-01-10T09:15:00Z",
        updatedAt: "2024-01-25T11:45:00Z",
        status: "published",
        executionCount: 123,
        lastRun: "2024-01-27T15:30:00Z",
      },
      {
        id: "3",
        name: "Security Incident Response",
        description:
          "Automated response to security threats with alerting and containment",
        category: "security",
        tags: ["security", "incident", "alerts"],
        author: "Security Team",
        createdAt: "2024-01-05T16:20:00Z",
        updatedAt: "2024-01-22T09:10:00Z",
        status: "published",
        executionCount: 12,
        lastRun: "2024-01-26T03:45:00Z",
      },
      {
        id: "4",
        name: "API Integration Testing",
        description:
          "Comprehensive testing suite for external API integrations",
        category: "testing",
        tags: ["testing", "api", "integration"],
        author: "QA Team",
        createdAt: "2024-01-12T13:30:00Z",
        updatedAt: "2024-01-24T16:20:00Z",
        status: "draft",
        executionCount: 8,
      },
      {
        id: "5",
        name: "Performance Monitoring Setup",
        description:
          "Deploy and configure application performance monitoring tools",
        category: "monitoring",
        tags: ["monitoring", "performance", "alerts"],
        author: "DevOps Team",
        createdAt: "2024-01-08T11:45:00Z",
        updatedAt: "2024-01-26T10:15:00Z",
        status: "published",
        executionCount: 67,
        lastRun: "2024-01-27T12:00:00Z",
      },
    ];

    setTimeout(() => {
      setPlaybooks(mockPlaybooks);
      setLoading(false);
    }, 1000);
  }, []);

  // Derived data
  const categories = [
    "all",
    ...Array.from(new Set(playbooks.map((p) => p.category))),
  ];

  const filteredPlaybooks = playbooks.filter((playbook) => {
    const matchesSearch =
      playbook.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      playbook.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      playbook.tags.some((tag) =>
        tag.toLowerCase().includes(searchTerm.toLowerCase()),
      );
    const matchesCategory =
      selectedCategory === "all" || playbook.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  // Handlers
  const handleCreatePlaybook = () => {
    router.push("/playbooks/editor");
  };

  const handleEditPlaybook = (playbook: Playbook) => {
    router.push(`/playbooks/editor?id=${playbook.id}`);
  };

  const handleExecutePlaybook = (playbook: Playbook) => {
    router.push(`/results?playbookId=${playbook.id}&action=execute`);
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: "user",
      content: currentMessage,
      timestamp: new Date().toISOString(),
    };

    setChatMessages((prev) => [...prev, userMessage]);
    setCurrentMessage("");
    setChatLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        "I can help you create a new playbook for that task. Let me break it down into steps...",
        "That's a great use case! I'll design a workflow that handles that efficiently.",
        "I understand. Let me create a playbook that automates that process for you.",
        "Perfect! I can set up monitoring and alerts for that scenario.",
        "Let me analyze your requirements and suggest the best approach...",
      ];

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content:
          responses[Math.floor(Math.random() * responses.length)] ||
          "I'm here to help you create playbooks. What would you like to build?",
        timestamp: new Date().toISOString(),
      };

      setChatMessages((prev) => [...prev, assistantMessage]);
      setChatLoading(false);
    }, 1500);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "published":
        return "bg-neon-green/20 text-neon-green border-neon-green/30";
      case "draft":
        return "bg-neon-yellow/20 text-neon-yellow border-neon-yellow/30";
      case "archived":
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
      default:
        return "bg-neon-cyan/20 text-neon-cyan border-neon-cyan/30";
    }
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      automation: "text-neon-cyan",
      workflow: "text-neon-magenta",
      security: "text-neon-red",
      testing: "text-neon-yellow",
      monitoring: "text-neon-green",
    };
    return colors[category as keyof typeof colors] || "text-neon-cyan";
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-cyber-dark flex items-center justify-center">
        <div className="flex items-center gap-3 text-neon-cyan">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span>Loading playbooks...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cyber-dark">
      {/* Header */}
      <div className="border-b border-neon-cyan/30 bg-black/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-neon-cyan mb-2">
                Playbooks
              </h1>
              <p className="text-neon-cyan/70">
                Create, manage, and execute automation workflows
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                onClick={() => setShowChat(!showChat)}
                className="text-neon-magenta border-neon-magenta hover:bg-neon-magenta/10"
              >
                <Bot className="h-4 w-4 mr-2" />
                Super Agent
              </Button>
              <Button
                onClick={handleCreatePlaybook}
                className="btn-neon text-neon-cyan"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Playbook
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6">
        <div className="flex gap-6">
          {/* Main Content */}
          <div className={cn("flex-1", showChat && "lg:pr-6")}>
            {/* Filters and Search */}
            <div className="flex flex-col md:flex-row gap-4 mb-6">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neon-cyan/50 h-4 w-4" />
                  <Input
                    placeholder="Search playbooks..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-cyber-dark border-neon-cyan/30 text-white"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3">
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="px-3 py-2 bg-cyber-dark border border-neon-cyan/30 rounded-md text-white"
                >
                  {categories.map((category) => (
                    <option key={category} value={category}>
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                    </option>
                  ))}
                </select>

                <div className="flex items-center border border-neon-cyan/30 rounded-md">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setViewMode("grid")}
                    className={cn(
                      "text-neon-cyan hover:bg-neon-cyan/10",
                      viewMode === "grid" && "bg-neon-cyan/20",
                    )}
                  >
                    <Grid className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setViewMode("list")}
                    className={cn(
                      "text-neon-cyan hover:bg-neon-cyan/10",
                      viewMode === "list" && "bg-neon-cyan/20",
                    )}
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Playbooks Display */}
            {filteredPlaybooks.length === 0 ? (
              <Card className="bg-cyber-dark border-neon-cyan/30">
                <CardContent className="text-center py-12">
                  <Zap className="h-12 w-12 text-neon-cyan/50 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">
                    No playbooks found
                  </h3>
                  <p className="text-neon-cyan/70 mb-4">
                    {searchTerm || selectedCategory !== "all"
                      ? "Try adjusting your search or filters"
                      : "Get started by creating your first playbook"}
                  </p>
                  <Button
                    onClick={handleCreatePlaybook}
                    className="btn-neon text-neon-cyan"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Create Playbook
                  </Button>
                </CardContent>
              </Card>
            ) : viewMode === "grid" ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredPlaybooks.map((playbook) => (
                  <Card
                    key={playbook.id}
                    className="bg-cyber-dark border-neon-cyan/30 hover:shadow-neon-cyan/20 hover:shadow-lg transition-all duration-200 cursor-pointer group"
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <CardTitle className="text-white text-lg mb-2 group-hover:text-neon-cyan transition-colors">
                            {playbook.name}
                          </CardTitle>
                          <Badge
                            className={cn(
                              "text-xs border",
                              getStatusColor(playbook.status),
                            )}
                          >
                            {playbook.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditPlaybook(playbook);
                            }}
                            className="text-neon-cyan hover:bg-neon-cyan/10"
                          >
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleExecutePlaybook(playbook);
                            }}
                            className="text-neon-green hover:bg-neon-green/10"
                          >
                            <Play className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>

                    <CardContent>
                      <p className="text-neon-cyan/70 text-sm mb-4 line-clamp-2">
                        {playbook.description}
                      </p>

                      <div className="flex flex-wrap gap-1 mb-4">
                        {playbook.tags.slice(0, 3).map((tag) => (
                          <Badge
                            key={tag}
                            variant="outline"
                            className="text-xs text-neon-magenta border-neon-magenta/30"
                          >
                            {tag}
                          </Badge>
                        ))}
                        {playbook.tags.length > 3 && (
                          <Badge
                            variant="outline"
                            className="text-xs text-neon-cyan/50 border-neon-cyan/30"
                          >
                            +{playbook.tags.length - 3}
                          </Badge>
                        )}
                      </div>

                      <div className="space-y-2 text-xs text-neon-cyan/60">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1">
                            <User className="h-3 w-3" />
                            {playbook.author}
                          </span>
                          <span className={getCategoryColor(playbook.category)}>
                            {playbook.category}
                          </span>
                        </div>

                        <div className="flex items-center justify-between">
                          <span>Executions: {playbook.executionCount}</span>
                          {playbook.lastRun && (
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDate(playbook.lastRun)}
                            </span>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredPlaybooks.map((playbook) => (
                  <Card
                    key={playbook.id}
                    className="bg-cyber-dark border-neon-cyan/30 hover:shadow-neon-cyan/20 hover:shadow-lg transition-all duration-200 cursor-pointer group"
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-white font-semibold group-hover:text-neon-cyan transition-colors">
                              {playbook.name}
                            </h3>
                            <Badge
                              className={cn(
                                "text-xs border",
                                getStatusColor(playbook.status),
                              )}
                            >
                              {playbook.status}
                            </Badge>
                            <span
                              className={cn(
                                "text-xs font-medium",
                                getCategoryColor(playbook.category),
                              )}
                            >
                              {playbook.category}
                            </span>
                          </div>
                          <p className="text-neon-cyan/70 text-sm mb-2">
                            {playbook.description}
                          </p>
                          <div className="flex items-center gap-4 text-xs text-neon-cyan/60">
                            <span className="flex items-center gap-1">
                              <User className="h-3 w-3" />
                              {playbook.author}
                            </span>
                            <span>Executions: {playbook.executionCount}</span>
                            {playbook.lastRun && (
                              <span className="flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                Last run: {formatDate(playbook.lastRun)}{" "}
                                {formatTime(playbook.lastRun)}
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex flex-wrap gap-1">
                            {playbook.tags.slice(0, 2).map((tag) => (
                              <Badge
                                key={tag}
                                variant="outline"
                                className="text-xs text-neon-magenta border-neon-magenta/30"
                              >
                                {tag}
                              </Badge>
                            ))}
                          </div>
                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleEditPlaybook(playbook);
                              }}
                              className="text-neon-cyan hover:bg-neon-cyan/10"
                            >
                              <Edit className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleExecutePlaybook(playbook);
                              }}
                              className="text-neon-green hover:bg-neon-green/10"
                            >
                              <Play className="h-4 w-4" />
                            </Button>
                          </div>
                          <ChevronRight className="h-4 w-4 text-neon-cyan/50" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>

          {/* Super Agent Chat Panel */}
          {showChat && (
            <div className="w-96 border-l border-neon-cyan/30">
              <Card className="h-[calc(100vh-12rem)] bg-cyber-dark border-neon-magenta/30">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-neon-magenta flex items-center gap-2">
                      <Bot className="h-5 w-5" />
                      Super Agent
                    </CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowChat(false)}
                      className="text-neon-cyan hover:bg-neon-cyan/10"
                    >
                      ×
                    </Button>
                  </div>
                </CardHeader>

                <CardContent className="flex flex-col h-full p-0">
                  <ScrollArea className="flex-1 px-4">
                    <div className="space-y-4 pb-4">
                      {chatMessages.map((message) => (
                        <div
                          key={message.id}
                          className={cn(
                            "flex",
                            message.type === "user"
                              ? "justify-end"
                              : "justify-start",
                          )}
                        >
                          <div
                            className={cn(
                              "max-w-[80%] p-3 rounded-lg text-sm",
                              message.type === "user"
                                ? "bg-neon-cyan/20 text-white border border-neon-cyan/30"
                                : "bg-neon-magenta/20 text-white border border-neon-magenta/30",
                            )}
                          >
                            {message.content}
                          </div>
                        </div>
                      ))}
                      {chatLoading && (
                        <div className="flex justify-start">
                          <div className="bg-neon-magenta/20 text-white border border-neon-magenta/30 p-3 rounded-lg">
                            <Loader2 className="h-4 w-4 animate-spin" />
                          </div>
                        </div>
                      )}
                    </div>
                  </ScrollArea>

                  <div className="p-4 border-t border-neon-magenta/30">
                    <div className="flex gap-2">
                      <Textarea
                        placeholder="Ask me to create, modify, or execute a playbook..."
                        value={currentMessage}
                        onChange={(e) => setCurrentMessage(e.target.value)}
                        onKeyPress={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        disabled={chatLoading}
                        className="flex-1 min-h-[40px] max-h-[120px] resize-none bg-cyber-dark border-neon-magenta/30 text-white"
                      />
                      <Button
                        onClick={handleSendMessage}
                        disabled={!currentMessage.trim() || chatLoading}
                        className="btn-neon text-neon-magenta self-end"
                      >
                        <Send className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function PlaybooksPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-cyber-dark flex items-center justify-center">
          <div className="flex items-center gap-3 text-neon-cyan">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span>Loading playbooks...</span>
          </div>
        </div>
      }
    >
      <PlaybooksContent />
    </Suspense>
  );
}
