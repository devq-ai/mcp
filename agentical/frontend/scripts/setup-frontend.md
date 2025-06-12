# Frontend Setup Guide

## Quick Setup Commands

Run these commands to set up the essential UI components and complete the MVP frontend:

```bash
# Navigate to frontend directory
cd agentical/frontend

# Install dependencies
npm install

# Install additional required packages
npm install @radix-ui/react-dialog @radix-ui/react-select @radix-ui/react-progress @radix-ui/react-badge @radix-ui/react-card @radix-ui/react-separator @radix-ui/react-scroll-area

# Install shadcn-ui CLI globally
npm install -g shadcn-ui

# Initialize shadcn-ui
npx shadcn-ui@latest init --yes

# Add essential UI components
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
npx shadcn-ui@latest add input
npx shadcn-ui@latest add label
npx shadcn-ui@latest add textarea
npx shadcn-ui@latest add select
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add badge
npx shadcn-ui@latest add progress
npx shadcn-ui@latest add separator
npx shadcn-ui@latest add scroll-area
npx shadcn-ui@latest add avatar
npx shadcn-ui@latest add skeleton

# Install development dependencies
npm install --save-dev @types/react @types/react-dom eslint-config-prettier prettier-plugin-tailwindcss

# Create missing essential files
mkdir -p components/layout components/chat components/playbook components/execution
mkdir -p types hooks stores
mkdir -p app/playbooks app/results
```

## Essential Files to Create

### 1. Component Files

Create these component files manually:

**components/providers.tsx** - App providers for TanStack Query, etc.
**components/theme-provider.tsx** - Theme provider for dark/light mode
**components/layout/navigation.tsx** - Main navigation component
**components/chat/chat-interface.tsx** - Chat interface for super agent
**components/playbook/playbook-card.tsx** - Playbook display component
**components/execution/execution-monitor.tsx** - Real-time execution monitoring

### 2. Page Files

**app/playbooks/page.tsx** - Playbook management page
**app/results/[id]/page.tsx** - Execution results page

### 3. API and Types

**lib/api.ts** - API client configuration
**types/api.ts** - TypeScript API types
**hooks/use-api.ts** - Custom API hooks

## Manual File Creation

Since we need to create these files manually, here are the essential ones:

### Missing Config Files

1. **components.json** (for shadcn-ui)
2. **postcss.config.js**
3. **tailwind.config.js** (update existing .ts file)

### Critical Components

1. **Navigation Component** - Header with menu
2. **Chat Interface** - For super agent interaction
3. **Playbook Card** - Display playbook information
4. **Execution Monitor** - Real-time progress tracking

## Development Workflow

1. **Setup Phase**: Run the commands above
2. **Component Phase**: Create missing components
3. **Page Phase**: Build the 3 main pages
4. **Integration Phase**: Connect with backend API
5. **Testing Phase**: Test user flows

## File Structure Check

After setup, your structure should look like:

```
frontend/
├── app/
│   ├── globals.css ✓
│   ├── layout.tsx ✓
│   ├── page.tsx ✓ (Dashboard)
│   ├── playbooks/
│   │   └── page.tsx (Playbook management)
│   └── results/
│       └── [id]/
│           └── page.tsx (Execution results)
├── components/
│   ├── ui/ (Shadcn components)
│   ├── layout/
│   ├── chat/
│   ├── playbook/
│   └── execution/
├── lib/
│   ├── utils.ts ✓
│   └── api.ts
├── types/
├── hooks/
└── stores/
```

## Next Steps

1. Run the setup commands
2. Create the missing components
3. Build the playbook and results pages
4. Connect to the backend API
5. Test the complete user flow

## MVP User Flow

1. **Dashboard** → View overview and recent playbooks
2. **Playbooks** → Select existing or create new via chat
3. **Results** → Monitor execution and view reports

This creates a focused, functional frontend that covers all the essential MVP requirements!