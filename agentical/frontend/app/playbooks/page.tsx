'use client';

import * as React from 'react';
import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { Plus, MessageCircle, Play, Search, Filter, Grid, List } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Playbook {
  id: string;
  name: string;
  description: string;
  category: string;
  status: 'active' | 'inactive' | 'draft';
  complexity_score: number;
  success_rate: number;
  avg_execution_time: number;
  last_run?: string;
  created_at: string;
  tags: string[];
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export default function PlaybooksPage() {
  const searchParams = useSearchParams();
  const action = searchParams.get('action');
  const playbookId = searchParams.get('id');

  const [mode, setMode] = useState<'select' | 'create'>('select');
  const [viewType, setViewType] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [playbooks, setPlaybooks] = useState<Playbook[]>([]);
  const [filteredPlaybooks, setFilteredPlaybooks] = useState<Playbook[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPlaybook, setSelectedPlaybook] = useState<string | null>(null);

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    if (action === 'create') {
      setMode('create');
    }
  }, [action]);

  useEffect(() => {
    // Mock data - replace with actual API call
    const mockPlaybooks: Playbook[] = [
      {
        id: '1',
        name: 'Data Processing Pipeline',
        description: 'Automated data ingestion, transformation, and validation workflow',
        category: 'data_processing',
        status: 'active',
        complexity_score: 7,
        success_rate: 95,
        avg_execution_time: 145,
        last_run: '2 hours ago',
        created_at: '2025-01-15',
        tags: ['automation', 'etl', 'validation'],
      },
      {
        id: '2',
        name: 'API Integration Suite',
        description: 'Connect and synchronize data between multiple external APIs',
        category: 'integration',
        status: 'active',
        complexity_score: 6,
        success_rate: 88,
        avg_execution_time: 230,
        last_run: '30 minutes ago',
        created_at: '2025-01-10',
        tags: ['api', 'sync', 'external'],
      },
      {
        id: '3',
        name: 'Report Generation',
        description: 'Generate comprehensive analytics reports with charts and insights',
        category: 'analytics',
        status: 'active',
        complexity_score: 5,
        success_rate: 92,
        avg_execution_time: 67,
        last_run: '1 day ago',
        created_at: '2025-01-08',
        tags: ['reports', 'analytics', 'charts'],
      },
      {
        id: '4',
        name: 'Database Backup',
        description: 'Automated database backup with compression and validation',
        category: 'automation',
        status: 'active',
        complexity_score: 4,
        success_rate: 78,
        avg_execution_time: 89,
        last_run: '3 hours ago',
        created_at: '2025-01-05',
        tags: ['backup', 'database', 'validation'],
      },
      {
        id: '5',
        name: 'Machine Learning Pipeline',
        description: 'End-to-end ML workflow with training, validation, and deployment',
        category: 'analytics',
        status: 'draft',
        complexity_score: 9,
        success_rate: 0,
        avg_execution_time: 0,
        created_at: '2025-01-28',
        tags: ['ml', 'training', 'deployment'],
      },
    ];

    setTimeout(() => {
      setPlaybooks(mockPlaybooks);
      setFilteredPlaybooks(mockPlaybooks);
      setLoading(false);
    }, 1000);
  }, []);

  useEffect(() => {
    let filtered = playbooks;

    // Filter by search query
    if (searchQuery) {
      filtered = filtered.filter(
        (playbook) =>
          playbook.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          playbook.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
          playbook.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }

    // Filter by category
    if (selectedCategory && selectedCategory !== 'all') {
      filtered = filtered.filter((playbook) => playbook.category === selectedCategory);
    }

    setFilteredPlaybooks(filtered);
  }, [playbooks, searchQuery, selectedCategory]);

  const categories = [
    { value: 'all', label: 'All Categories', count: playbooks.length },
    { value: 'automation', label: 'Automation', count: playbooks.filter(p => p.category === 'automation').length },
    { value: 'data_processing', label: 'Data Processing', count: playbooks.filter(p => p.category === 'data_processing').length },
    { value: 'integration', label: 'Integration', count: playbooks.filter(p => p.category === 'integration').length },
    { value: 'analytics', label: 'Analytics', count: playbooks.filter(p => p.category === 'analytics').length },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'status-done';
      case 'inactive': return 'status-backlog';
      case 'draft': return 'status-planning';
      default: return 'status-backlog';
    }
  };

  const getComplexityColor = (score: number) => {
    if (score <= 3) return 'text-neon-lime';
    if (score <= 6) return 'text-neon-orange';
    return 'text-neon-red';
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setChatLoading(true);

    // Simulate super agent response
    setTimeout(() => {
      const responses = [
        "I understand you want to create a new playbook. Could you tell me more about what you're trying to accomplish?",
        "That sounds like an interesting automation challenge. What kind of data sources will you be working with?",
        "Great! I'll help you design a playbook for that. Let me break this down into manageable steps...",
        "Based on your requirements, I recommend using the following agents: super_agent for coordination, codifier for any code generation, and io for data handling. Does this sound right?",
        "Perfect! I'm creating a playbook structure for you. This will include data validation, processing steps, and error handling. Would you like me to add any specific requirements?",
      ];

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responses[Math.floor(Math.random() * responses.length)],
        timestamp: new Date().toISOString(),
      };

      setChatMessages(prev => [...prev, assistantMessage]);
      setChatLoading(false);
    }, 1500);
  };

  const handleExecutePlaybook = (playbookId: string) => {
    // Navigate to results page
    window.location.href = `/results/${playbookId}`;
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-gradient">Playbooks</h1>
          <p className="text-xl text-muted-foreground">
            {mode === 'select' ? 'Select existing or create new playbooks' : 'Chat with Super Agent'}
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <Button
            variant={mode === 'select' ? 'default' : 'outline'}
            onClick={() => setMode('select')}
            className={mode === 'select' ? 'btn-neon text-neon-lime' : ''}
          >
            <Grid className="h-4 w-4 mr-2" />
            Browse
          </Button>
          <Button
            variant={mode === 'create' ? 'default' : 'outline'}
            onClick={() => setMode('create')}
            className={mode === 'create' ? 'btn-neon text-neon-magenta' : ''}
          >
            <Plus className="h-4 w-4 mr-2" />
            Create New
          </Button>
        </div>
      </div>

      {mode === 'select' ? (
        <>
          {/* Search and Filters */}
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search playbooks..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Category Filter */}
              <div className="flex flex-wrap gap-2">
                {categories.map((category) => (
                  <Button
                    key={category.value}
                    variant={selectedCategory === category.value ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedCategory(category.value)}
                    className={cn(
                      'text-xs',
                      selectedCategory === category.value && 'btn-neon text-neon-lime'
                    )}
                  >
                    {category.label} ({category.count})
                  </Button>
                ))}
              </div>

              {/* View Toggle */}
              <div className="flex items-center border rounded-lg">
                <Button
                  variant={viewType === 'grid' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewType('grid')}
                  className="rounded-r-none"
                >
                  <Grid className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewType === 'list' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewType('list')}
                  className="rounded-l-none"
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>

          {/* Playbooks Grid/List */}
          <div className={cn(
            'grid gap-6',
            viewType === 'grid'
              ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
              : 'grid-cols-1'
          )}>
            {filteredPlaybooks.map((playbook) => (
              <Card
                key={playbook.id}
                className={cn(
                  'cyber-card cursor-pointer transition-all',
                  selectedPlaybook === playbook.id && 'border-primary shadow-neon'
                )}
                onClick={() => setSelectedPlaybook(selectedPlaybook === playbook.id ? null : playbook.id)}
              >
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="text-lg">{playbook.name}</CardTitle>
                      <CardDescription className="mt-1">
                        {playbook.description}
                      </CardDescription>
                    </div>
                    <Badge className={getStatusColor(playbook.status)}>
                      {playbook.status}
                    </Badge>
                  </div>
                </CardHeader>

                <CardContent>
                  <div className="space-y-4">
                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Complexity</div>
                        <div className={cn('font-medium', getComplexityColor(playbook.complexity_score))}>
                          {playbook.complexity_score}/10
                        </div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Success</div>
                        <div className="font-medium text-neon-lime">
                          {playbook.success_rate}%
                        </div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Avg Time</div>
                        <div className="font-medium">
                          {playbook.avg_execution_time}s
                        </div>
                      </div>
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1">
                      {playbook.tags.map((tag) => (
                        <Badge key={tag} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>

                    {/* Last Run */}
                    {playbook.last_run && (
                      <div className="text-xs text-muted-foreground">
                        Last run: {playbook.last_run}
                      </div>
                    )}

                    {/* Actions */}
                    {selectedPlaybook === playbook.id && (
                      <div className="flex space-x-2 pt-2 border-t border-border/50">
                        <Button
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleExecutePlaybook(playbook.id);
                          }}
                          className="flex-1 btn-neon text-neon-magenta"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Execute
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation();
                            // Handle view details
                          }}
                        >
                          Details
                        </Button>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {filteredPlaybooks.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground mb-4">No playbooks found</p>
              <Button
                onClick={() => setMode('create')}
                className="btn-neon text-neon-magenta"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Your First Playbook
              </Button>
            </div>
          )}
        </>
      ) : (
        /* Chat Interface */
        <div className="max-w-4xl mx-auto">
          <Card className="cyber-card h-[600px] flex flex-col">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageCircle className="h-5 w-5 text-neon-magenta" />
                Chat with Super Agent
              </CardTitle>
              <CardDescription>
                Describe what you want to accomplish and I'll help you create the perfect playbook
              </CardDescription>
            </CardHeader>

            {/* Chat Messages */}
            <CardContent className="flex-1 flex flex-col">
              <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                {chatMessages.length === 0 && (
                  <div className="text-center text-muted-foreground py-8">
                    <MessageCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Start a conversation to create your playbook</p>
                    <p className="text-sm mt-2">Try: "I need to process data from multiple APIs and generate reports"</p>
                  </div>
                )}

                {chatMessages.map((message) => (
                  <div
                    key={message.id}
                    className={cn(
                      'flex',
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    )}
                  >
                    <div
                      className={cn(
                        'max-w-[80%] rounded-lg px-4 py-2 border',
                        message.role === 'user'
                          ? 'chat-bubble-user'
                          : 'chat-bubble-agent'
                      )}
                    >
                      <p className="text-sm">{message.content}</p>
                      <span className="text-xs opacity-70 mt-1 block">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))}

                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="chat-bubble-agent max-w-[80%]">
                      <div className="flex items-center space-x-2">
                        <div className="loading-spinner w-4 h-4"></div>
                        <span className="text-sm">Super Agent is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <div className="flex space-x-2">
                <Input
                  placeholder="Describe what you want to accomplish..."
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  disabled={chatLoading}
                  className="flex-1"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!currentMessage.trim() || chatLoading}
                  className="btn-neon text-neon-magenta"
                >
                  Send
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
