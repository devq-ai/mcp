// Global type definitions for Agentical Frontend

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer';
  created_at: string;
  last_login?: string;
}

export interface Agent {
  id: string;
  name: string;
  type: 'super_agent' | 'playbook_agent' | 'codifier' | 'io' | 'custom';
  status: 'active' | 'inactive' | 'busy' | 'error';
  capabilities: string[];
  configuration: Record<string, any>;
  performance_metrics: {
    success_rate: number;
    avg_response_time: number;
    total_executions: number;
  };
  created_at: string;
  updated_at: string;
}

export interface Playbook {
  id: string;
  name: string;
  description: string;
  category: PlaybookCategory;
  status: PlaybookStatus;
  complexity_score: number;
  estimated_duration: number;
  success_rate: number;
  steps: PlaybookStep[];
  variables: PlaybookVariable[];
  agents: string[]; // Agent IDs
  created_at: string;
  updated_at: string;
  created_by: string;
  tags: string[];
}

export interface PlaybookStep {
  id: string;
  name: string;
  description: string;
  step_type: PlaybookStepType;
  order: number;
  agent_id: string;
  configuration: StepConfiguration;
  dependencies: string[]; // Step IDs this step depends on
  estimated_duration: number;
  retry_config?: RetryConfiguration;
}

export interface PlaybookVariable {
  id: string;
  name: string;
  type: VariableType;
  required: boolean;
  default_value?: any;
  description?: string;
  validation_rules?: ValidationRule[];
}

export interface PlaybookExecution {
  id: string;
  playbook_id: string;
  playbook_name: string;
  status: ExecutionStatus;
  progress_percentage: number;
  current_step?: string;
  current_step_name?: string;
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  input_variables: Record<string, any>;
  output_data?: any;
  error_message?: string;
  step_executions: StepExecution[];
  logs: ExecutionLog[];
  created_by: string;
  execution_mode: ExecutionMode;
}

export interface StepExecution {
  id: string;
  step_id: string;
  step_name: string;
  agent_id: string;
  status: StepExecutionStatus;
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  input_data?: any;
  output_data?: any;
  error_message?: string;
  retry_count: number;
}

export interface ExecutionLog {
  id: string;
  execution_id: string;
  timestamp: string;
  level: LogLevel;
  message: string;
  step_id?: string;
  step_name?: string;
  agent_id?: string;
  agent_name?: string;
  metadata?: Record<string, any>;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  context: ChatContext;
  status: 'active' | 'completed' | 'archived';
  created_at: string;
  updated_at: string;
  playbook_id?: string; // If session results in a playbook
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  metadata?: MessageMetadata;
  attachments?: MessageAttachment[];
}

export interface ChatContext {
  user_intent?: string;
  domain?: string;
  complexity_level?: 'simple' | 'medium' | 'complex';
  requirements?: string[];
  constraints?: string[];
  preferred_agents?: string[];
}

export interface MessageMetadata {
  typing_time?: number;
  confidence_score?: number;
  suggested_actions?: SuggestedAction[];
  playbook_preview?: PlaybookPreview;
}

export interface MessageAttachment {
  id: string;
  type: 'file' | 'image' | 'code' | 'config';
  name: string;
  size: number;
  url: string;
  mime_type: string;
}

export interface SuggestedAction {
  id: string;
  type: 'create_playbook' | 'run_existing' | 'modify_playbook' | 'ask_question';
  label: string;
  description: string;
  payload?: any;
}

export interface PlaybookPreview {
  name: string;
  description: string;
  estimated_steps: number;
  estimated_duration: number;
  required_agents: string[];
  complexity_score: number;
}

export interface SystemMetrics {
  total_playbooks: number;
  active_executions: number;
  total_executions_today: number;
  success_rate_24h: number;
  avg_execution_time: number;
  system_load: number;
  agent_status: Record<string, AgentStatus>;
  resource_usage: ResourceUsage;
}

export interface AgentStatus {
  status: 'online' | 'offline' | 'busy' | 'error';
  current_tasks: number;
  queue_length: number;
  last_heartbeat: string;
  performance: {
    success_rate: number;
    avg_response_time: number;
    error_count_24h: number;
  };
}

export interface ResourceUsage {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    bytes_in: number;
    bytes_out: number;
  };
  database_connections: number;
}

export interface AnalyticsData {
  executions_over_time: TimeSeriesData[];
  success_rate_trend: TimeSeriesData[];
  popular_playbooks: PlaybookPopularity[];
  agent_performance: AgentPerformanceData[];
  error_breakdown: ErrorBreakdown[];
}

export interface TimeSeriesData {
  timestamp: string;
  value: number;
  label?: string;
}

export interface PlaybookPopularity {
  playbook_id: string;
  playbook_name: string;
  execution_count: number;
  success_rate: number;
  avg_duration: number;
}

export interface AgentPerformanceData {
  agent_id: string;
  agent_name: string;
  total_executions: number;
  success_rate: number;
  avg_response_time: number;
  error_count: number;
}

export interface ErrorBreakdown {
  error_type: string;
  count: number;
  percentage: number;
  recent_examples: string[];
}

// Enum-like types
export type PlaybookCategory =
  | 'automation'
  | 'data_processing'
  | 'integration'
  | 'analytics'
  | 'monitoring'
  | 'custom';

export type PlaybookStatus =
  | 'draft'
  | 'active'
  | 'inactive'
  | 'archived'
  | 'deprecated';

export type PlaybookStepType =
  | 'action'
  | 'conditional'
  | 'loop'
  | 'parallel'
  | 'wait'
  | 'notification';

export type VariableType =
  | 'string'
  | 'number'
  | 'boolean'
  | 'object'
  | 'array'
  | 'file'
  | 'secret';

export type ExecutionStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'timeout';

export type StepExecutionStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'skipped'
  | 'retrying';

export type LogLevel =
  | 'debug'
  | 'info'
  | 'warning'
  | 'error'
  | 'critical';

export type MessageRole =
  | 'user'
  | 'assistant'
  | 'system'
  | 'agent';

export type ExecutionMode =
  | 'sequential'
  | 'parallel'
  | 'hybrid';

// Configuration types
export interface StepConfiguration {
  timeout_seconds?: number;
  max_retries?: number;
  retry_delay_seconds?: number;
  parallel_execution?: boolean;
  required_resources?: string[];
  environment_variables?: Record<string, string>;
  custom_parameters?: Record<string, any>;
}

export interface RetryConfiguration {
  max_attempts: number;
  initial_delay_seconds: number;
  max_delay_seconds: number;
  backoff_multiplier: number;
  retry_on_errors: string[];
}

export interface ValidationRule {
  type: 'required' | 'min_length' | 'max_length' | 'pattern' | 'custom';
  value?: any;
  message: string;
}

// UI State types
export interface AppState {
  user?: User;
  theme: 'light' | 'dark';
  sidebar_collapsed: boolean;
  active_execution?: string;
  notifications: Notification[];
  settings: AppSettings;
}

export interface AppSettings {
  auto_refresh_interval: number;
  show_debug_logs: boolean;
  notification_preferences: NotificationPreferences;
  display_preferences: DisplayPreferences;
}

export interface NotificationPreferences {
  execution_started: boolean;
  execution_completed: boolean;
  execution_failed: boolean;
  system_alerts: boolean;
  agent_offline: boolean;
}

export interface DisplayPreferences {
  compact_mode: boolean;
  show_timestamps: boolean;
  group_logs_by_step: boolean;
  highlight_errors: boolean;
  auto_scroll_logs: boolean;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  label: string;
  action: string;
  variant?: 'default' | 'destructive';
}

// API Response types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  status: number;
  timestamp: string;
}

export interface ApiError {
  message: string;
  status?: number;
  code?: string;
  details?: any;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

// WebSocket types
export interface WebSocketMessage {
  type: 'execution_update' | 'agent_status' | 'system_alert' | 'chat_message';
  payload: any;
  timestamp: string;
  id?: string;
}

export interface ExecutionUpdate {
  execution_id: string;
  status: ExecutionStatus;
  progress_percentage: number;
  current_step?: string;
  new_logs?: ExecutionLog[];
  step_updates?: StepExecution[];
}

// Form types
export interface PlaybookFormData {
  name: string;
  description: string;
  category: PlaybookCategory;
  tags: string[];
  variables: PlaybookVariable[];
  steps: Omit<PlaybookStep, 'id'>[];
}

export interface ExecutionFormData {
  playbook_id: string;
  execution_mode: ExecutionMode;
  variables: Record<string, any>;
  timeout_minutes?: number;
  notification_preferences?: {
    on_completion: boolean;
    on_failure: boolean;
  };
}

// Component Props types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingState {
  isLoading: boolean;
  error?: string | null;
}

export interface TableColumn<T> {
  key: keyof T | string;
  label: string;
  sortable?: boolean;
  render?: (value: any, item: T) => React.ReactNode;
  width?: string;
}

export interface FilterOption {
  value: string;
  label: string;
  count?: number;
}

export interface SortOption {
  value: string;
  label: string;
  direction: 'asc' | 'desc';
}

// Utility types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;
