import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { cn } from '@/lib/utils';
import { Providers } from '@/components/providers';
import { Navigation } from '@/components/layout/navigation';
import { ThemeProvider } from '@/components/theme-provider';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
});

export const metadata: Metadata = {
  title: {
    default: 'Agentical - AI Playbook Execution Platform',
    template: '%s | Agentical',
  },
  description: 'Create, execute, and monitor AI-powered playbooks with real-time agent orchestration.',
  keywords: [
    'AI',
    'automation',
    'playbooks',
    'agents',
    'workflow',
    'orchestration',
    'DevQ.ai',
  ],
  authors: [
    {
      name: 'DevQ.ai Team',
      url: 'https://github.com/devq-ai',
    },
  ],
  creator: 'DevQ.ai',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://agentical.devq.ai',
    title: 'Agentical - AI Playbook Execution Platform',
    description: 'Create, execute, and monitor AI-powered playbooks with real-time agent orchestration.',
    siteName: 'Agentical',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Agentical - AI Playbook Execution Platform',
    description: 'Create, execute, and monitor AI-powered playbooks with real-time agent orchestration.',
    creator: '@devqai',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          'min-h-screen bg-background font-sans antialiased cyber-bg',
          inter.variable,
          jetbrainsMono.variable
        )}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <Providers>
            <div className="relative flex min-h-screen flex-col">
              {/* Navigation */}
              <Navigation />

              {/* Main Content */}
              <main className="flex-1">
                {children}
              </main>

              {/* Footer */}
              <footer className="border-t border-border/50 bg-card/30 backdrop-blur-sm">
                <div className="container flex h-14 items-center justify-between px-4">
                  <div className="flex items-center space-x-4">
                    <p className="text-sm text-muted-foreground">
                      © 2025 DevQ.ai - All Rights Reserved
                    </p>
                  </div>
                  <div className="flex items-center space-x-4">
                    <a
                      href="https://github.com/devq-ai/agentical"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-muted-foreground hover:text-primary transition-colors"
                    >
                      GitHub
                    </a>
                    <a
                      href="https://docs.devq.ai"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-muted-foreground hover:text-primary transition-colors"
                    >
                      Documentation
                    </a>
                  </div>
                </div>
              </footer>
            </div>
          </Providers>
        </ThemeProvider>
      </body>
    </html>
  );
}
