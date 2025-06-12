# Agentical Frontend - MVP

A streamlined Next.js frontend for the Agentical playbook execution system.

## 🎯 MVP Features

### Three Core Pages
1. **Dashboard** (`/`) - Main landing page with overview
2. **Playbooks** (`/playbooks`) - Select existing or create new playbooks 
3. **Results** (`/results/[id]`) - Real-time execution monitoring and reports

### User Flow
```
Dashboard → Select/Create Playbook → Execute → Monitor Results
```

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open http://localhost:3000
```

## 📁 Project Structure

```
frontend/
├── app/                     # Next.js 14 App Router
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Dashboard
│   ├── playbooks/          # Playbook management
│   │   └── page.tsx
│   └── results/            # Execution monitoring
│       └── [id]/
│           └── page.tsx
├── components/             # Reusable components
│   ├── ui/                # Shadcn UI components
│   ├── chat/              # Chat interface
│   ├── playbook/          # Playbook components
│   └── execution/         # Execution monitoring
├── lib/                   # Utilities
│   ├── api.ts            # API client
│   ├── websocket.ts      # Real-time connections
│   └── utils.ts          # Helper functions
├── types/                 # TypeScript definitions
└── styles/               # Global styles
```

## 🎨 Design System

### Color Palette (Cyber Dark Theme)
- **Primary**: #FF0090 (Neon Magenta)
- **Secondary**: #C7EA46 (Neon Lime)
- **Accent**: #FF5F1F (Neon Orange)
- **Background**: #0A0A0A (Pure Black)
- **Surface**: #2C2F33 (Gunmetal Grey)

### Status Colors
- **Backlog**: #A5A5A5 (Pastel Gray)
- **Planning**: #74C3D1 (Pastel Cyan)
- **Active**: #A1D9A0 (Pastel Green)
- **Complete**: #B5A0E3 (Pastel Purple)

## 🔧 Technology Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Components**: Shadcn UI
- **API Client**: TanStack Query
- **State**: Zustand
- **Real-time**: WebSocket
- **Testing**: Vitest

## 📡 API Integration

### Endpoints Used
- `GET /v1/playbooks` - List existing playbooks
- `POST /v1/playbooks` - Create new playbook
- `POST /v1/playbooks/{id}/execute` - Start execution
- `GET /v1/playbooks/{id}/executions/{execution_id}` - Get execution status
- `WS /v1/playbooks/executions/{id}/stream` - Real-time updates

### Agent Communication
- **super_agent** - Main user interface agent
- **playbook_agent** - Creates playbook structure
- **codifier** - Code generation and validation
- **io** - Input/output handling

## 🎮 User Experience

### Page 1: Dashboard
```
┌─────────────────────────────┐
│  🏠 Agentical Dashboard      │
├─────────────────────────────┤
│  Recent Playbooks           │
│  ┌─────┐ ┌─────┐ ┌─────┐    │
│  │ P1  │ │ P2  │ │ P3  │    │
│  └─────┘ └─────┘ └─────┘    │
│                             │
│  [Create New Playbook]      │
│  [View All Playbooks]       │
└─────────────────────────────┘
```

### Page 2: Playbooks
```
┌─────────────────────────────┐
│  📋 Playbook Management     │
├─────────────────────────────┤
│  Existing Playbooks         │
│  ┌─────────────────────────┐ │
│  │ □ Data Processing       │ │
│  │ □ API Integration       │ │
│  │ □ Report Generation     │ │
│  └─────────────────────────┘ │
│                             │
│  Create New Playbook        │
│  ┌─────────────────────────┐ │
│  │ 💬 Chat with Super Agent│ │
│  │ User: "I need to..."    │ │
│  │ Super: "I'll help..."   │ │
│  └─────────────────────────┘ │
│                             │
│  [Start Selected Playbook]  │
└─────────────────────────────┘
```

### Page 3: Results
```
┌─────────────────────────────┐
│  ⚡ Execution Monitor       │
├─────────────────────────────┤
│  Visual Progress            │
│  ┌─────────────────────────┐ │
│  │ [●]──[●]──[ ]──[ ]     │ │
│  │ Step1 Step2 Step3 Step4 │ │
│  └─────────────────────────┘ │
│                             │
│  Live Logs                  │
│  ┌─────────────────────────┐ │
│  │ [12:34] Starting step 1 │ │
│  │ [12:35] Processing...   │ │
│  │ [12:36] Step 1 complete │ │
│  └─────────────────────────┘ │
│                             │
│  📊 [View Report] (when done)│
└─────────────────────────────┘
```

## 🔄 Development Workflow

1. **Setup**: Initialize Next.js project with Shadcn UI
2. **Layout**: Create responsive layout with navigation
3. **Dashboard**: Build overview page with recent playbooks
4. **Playbooks**: Implement selection and chat interface
5. **Results**: Add real-time monitoring and logs
6. **Polish**: Add animations and error handling

## 🧪 Testing Strategy

- **Unit**: Component testing with Vitest
- **Integration**: API integration tests
- **E2E**: User flow testing with Playwright
- **Real-time**: WebSocket connection testing

## 🚀 Deployment

- **Development**: Local development server
- **Staging**: Vercel preview deployments
- **Production**: Vercel production deployment

---

**MVP Goal**: Simple, focused UI that enables users to create, select, execute, and monitor playbooks through an intuitive chat-driven interface.