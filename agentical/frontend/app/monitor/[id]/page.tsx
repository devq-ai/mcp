"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ExternalLink } from "lucide-react";
import ExecutionMonitor from "@/components/monitor/ExecutionMonitor";

interface ExecutionData {
  id: string;
  playbookName: string;
  status: "pending" | "running" | "completed" | "failed" | "paused" | "cancelled";
  startTime: string;
  endTime?: string;
}

export default function ExecutionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const executionId = params.id as string;

  const [executionData, setExecutionData] = useState<ExecutionData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Mock data - in real app, fetch from API
  useEffect(() => {
    const mockData: ExecutionData = {
      id: executionId,
      playbookName: "User Onboarding Flow",
      status: "running",
      startTime: new Date(Date.now() - 300000).toISOString(),
    };

    setTimeout(() => {
      setExecutionData(mockData);
      setIsLoading(false);
    }, 500);
  }, [executionId]);

  const handleStatusChange = (action: "pause" | "resume" | "stop" | "restart") => {
    console.log(`Action requested: ${action} for execution ${executionId}`);

    // In a real app, make API call here
    if (executionData) {
      let newStatus = executionData.status;

      switch (action) {
        case "pause":
          newStatus = "paused";
          break;
        case "resume":
          newStatus = "running";
          break;
        case "stop":
          newStatus = "cancelled";
          break;
        case "restart":
          newStatus = "running";
          break;
      }

      setExecutionData({
        ...executionData,
        status: newStatus,
      });
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-cyber-dark text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-neon-cyan"></div>
      </div>
    );
  }

  if (!executionData) {
    return (
      <div className="min-h-screen bg-cyber-dark text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-neon-red mb-4">Execution Not Found</h1>
          <p className="text-neon-cyan/70 mb-6">
            The execution with ID "{executionId}" could not be found.
          </p>
          <Button
            onClick={() => router.push("/monitor")}
            className="bg-neon-cyan text-cyber-dark hover:bg-neon-cyan/80"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Monitor
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cyber-dark">
      <div className="p-6">
        {/* Navigation */}
        <div className="flex items-center gap-4 mb-6">
          <Button
            onClick={() => router.push("/monitor")}
            variant="outline"
            size="sm"
            className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Monitor
          </Button>

          <div className="flex-1" />

          <Button
            onClick={() => router.push("/playbooks/editor")}
            variant="outline"
            size="sm"
            className="border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10"
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in Editor
          </Button>
        </div>

        {/* Execution Monitor */}
        <ExecutionMonitor
          executionId={executionData.id}
          playbookName={executionData.playbookName}
          status={executionData.status}
          onStatusChange={handleStatusChange}
        />
      </div>
    </div>
  );
}
