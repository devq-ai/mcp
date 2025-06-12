'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Play, Plus, BarChart3, Clock, CheckCircle2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface RecentPlaybook {
  id: string;
  name: string;
  description: string;
  status: 'completed' | 'running' | 'failed' | 'pending';
  lastRun: string;
  successRate: number;
  executionTime: number;
}

interface SystemStats {
  totalPlaybooks: number;
  activeExecutions: number;
  successRate: number;
  avgExecutionTime: number;
}

export default function Dashboard() {
  const [recentPlaybooks, setRecentPlaybooks] = useState<RecentPlaybook[]>([]);
  const [systemStats, setSystemStats] = useState<SystemStats>({
    totalPlaybooks: 0,
    activeExecutions: 0,
    successRate: 0,
    avgExecutionTime: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock data - replace with actual API calls
    const mockRecentPlaybooks: RecentPlaybook[] = [
      {
        id: '1',
        name: 'Data Processing Pipeline',
        description: 'Automated data ingestion and transformation workflow',
        status: 'completed',
        lastRun: '2 hours ago',
        successRate: 95,
        executionTime: 145,
      },
      {
        id: '2',
        name: 'API Integration Suite',
        description: 'Connect and synchronize multiple external APIs',
        status: 'running',
        lastRun: '30 minutes ago',
        successRate: 88,
        executionTime: 230,
      },
      {
        id: '3',
        name: 'Report Generation',
        description: 'Generate comprehensive analytics reports',
        status: 'completed',
        lastRun: '1 day ago',
        successRate: 92,
        executionTime: 67,
      },
      {
        id: '4',
        name: 'Database Backup',
        description: 'Automated database backup and validation',
        status: 'failed',
        lastRun: '3 hours ago',
        successRate: 78,
        executionTime: 89,
      },
    ];

    const mockSystemStats: SystemStats = {
      totalPlaybooks: 24,
      activeExecutions: 3,
      successRate: 89,
      avgExecutionTime: 156,
    };

    // Simulate loading
    setTimeout(() => {
      setRecentPlaybooks(mockRecentPlaybooks);
      setSystemStats(mockSystemStats);
      setLoading(false);
    }, 1000);
  }, []);

  const getStatusColor = (status: RecentPlaybook['status']) => {
    switch (status) {
      case 'completed':
        return 'status-done';
      case 'running':
        return 'status-doing';
      case 'failed':
        return 'status-tech-debt';
      case 'pending':
        return 'status-todo';
      default:
        return 'status-backlog';
    }
  };

  const getStatusIcon = (status: RecentPlaybook['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4" />;
      case 'running':
        return <Play className="h-4 w-4" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4" />;
      case 'pending':
        return <Clock className="h-4 w-4" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-8">
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="loading-spinner w-8 h-8 text-primary"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-8 space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold text-gradient">
          Agentical Dashboard
        </h1>
        <p className="text-xl text-muted-foreground">
          AI-powered playbook execution platform
        </p>
      </div>

      {/* System Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="cyber-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Playbooks</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-magenta">
              {systemStats.totalPlaybooks}
            </div>
            <p className="text-xs text-muted-foreground">
              Available for execution
            </p>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Executions</CardTitle>
            <Play className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-lime">
              {systemStats.activeExecutions}
            </div>
            <p className="text-xs text-muted-foreground">
              Currently running
            </p>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-green">
              {systemStats.successRate}%
            </div>
            <Progress value={systemStats.successRate} className="mt-2" />
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Execution Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-neon-orange">
              {systemStats.avgExecutionTime}s
            </div>
            <p className="text-xs text-muted-foreground">
              Average completion time
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5 text-neon-magenta" />
              Create New Playbook
            </CardTitle>
            <CardDescription>
              Chat with our super agent to design a custom playbook for your needs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/playbooks?action=create">
              <Button className="w-full btn-neon text-neon-magenta" variant="outline">
                Start Creating
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-neon-lime" />
              Browse Playbooks
            </CardTitle>
            <CardDescription>
              Select from existing playbooks or manage your playbook library
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/playbooks">
              <Button className="w-full btn-neon text-neon-lime" variant="outline">
                View Library
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* Recent Playbooks */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-neon-cyan" />
            Recent Playbooks
          </CardTitle>
          <CardDescription>
            Your recently accessed and executed playbooks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentPlaybooks.map((playbook) => (
              <div
                key={playbook.id}
                className="flex items-center justify-between p-4 rounded-lg border border-border/50 hover:border-primary/50 transition-all group"
              >
                <div className="flex items-center space-x-4">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-muted/20">
                    {getStatusIcon(playbook.status)}
                  </div>
                  <div>
                    <h3 className="font-semibold group-hover:text-primary transition-colors">
                      {playbook.name}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {playbook.description}
                    </p>
                    <div className="flex items-center space-x-4 mt-1">
                      <span className="text-xs text-muted-foreground">
                        Last run: {playbook.lastRun}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {playbook.executionTime}s avg
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <div className="text-right">
                    <Badge className={getStatusColor(playbook.status)}>
                      {playbook.status}
                    </Badge>
                    <div className="text-xs text-muted-foreground mt-1">
                      {playbook.successRate}% success
                    </div>
                  </div>

                  {playbook.status === 'running' ? (
                    <Link href={`/results/${playbook.id}`}>
                      <Button size="sm" variant="outline" className="btn-neon text-neon-lime">
                        Monitor
                      </Button>
                    </Link>
                  ) : (
                    <div className="flex space-x-2">
                      <Link href={`/playbooks?id=${playbook.id}`}>
                        <Button size="sm" variant="outline">
                          View
                        </Button>
                      </Link>
                      <Link href={`/playbooks?id=${playbook.id}&action=execute`}>
                        <Button size="sm" className="btn-neon text-neon-magenta">
                          Run
                        </Button>
                      </Link>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {recentPlaybooks.length === 0 && (
            <div className="text-center py-8">
              <p className="text-muted-foreground">No recent playbooks found</p>
              <Link href="/playbooks?action=create">
                <Button className="mt-4 btn-neon text-neon-magenta" variant="outline">
                  Create Your First Playbook
                </Button>
              </Link>
            </div>
          )}
        </CardContent>
      </Card>

      {/* System Status */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-neon-green animate-pulse"></div>
            System Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-muted/10">
              <span className="text-sm">API Service</span>
              <Badge className="status-done">Operational</Badge>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-muted/10">
              <span className="text-sm">Agent Network</span>
              <Badge className="status-done">Healthy</Badge>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-muted/10">
              <span className="text-sm">Execution Engine</span>
              <Badge className="status-done">Active</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
