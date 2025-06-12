#!/usr/bin/env python3
"""
Monitoring Tool Demo

This script demonstrates comprehensive usage of the Agentical Monitoring Tool,
showcasing system monitoring, alerting, custom metrics, and various export formats.

Usage:
    python monitoring_tool_demo.py

Features Demonstrated:
- System metrics monitoring
- Target health checks
- Custom metric creation
- Alert management
- Data export in multiple formats
- Real-time dashboard simulation
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.security.monitoring_tool import (
    MonitoringTool,
    MetricType,
    AlertLevel
)


class MonitoringDemo:
    """Demo class for monitoring tool functionality."""

    def __init__(self):
        """Initialize the monitoring demo."""
        self.config = {
            "check_interval": 10,  # Check every 10 seconds
            "alert_retention_days": 7,
            "max_targets": 50,
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_threshold": 90.0,
            "email_smtp_server": "smtp.gmail.com",
            "email_smtp_port": 587,
            "email_from": "monitoring@example.com",
            "email_to": ["admin@example.com"],
            "slack_webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "webhook_url": "https://your-webhook-endpoint.com/alerts"
        }

        self.monitoring_tool = MonitoringTool(self.config)
        self.demo_running = False

    async def setup_monitoring_targets(self):
        """Set up various monitoring targets for demonstration."""
        print("🎯 Setting up monitoring targets...")

        # Add web service targets
        targets = [
            {
                "name": "httpbin-status",
                "url": "https://httpbin.org/status/200",
                "check_interval": 30,
                "timeout": 10
            },
            {
                "name": "httpbin-delay",
                "url": "https://httpbin.org/delay/2",
                "check_interval": 60,
                "timeout": 15
            },
            {
                "name": "google-dns",
                "url": "https://8.8.8.8",
                "check_interval": 120,
                "timeout": 5
            },
            {
                "name": "local-service",
                "url": "http://localhost:8080/health",
                "check_interval": 30,
                "timeout": 5
            }
        ]

        for target in targets:
            try:
                await self.monitoring_tool.add_target(**target)
                print(f"  ✅ Added target: {target['name']}")
            except Exception as e:
                print(f"  ❌ Failed to add target {target['name']}: {e}")

    def setup_custom_metrics(self):
        """Set up custom metrics for demonstration."""
        print("📊 Setting up custom metrics...")

        # Application performance metrics
        def app_response_time_collector():
            """Simulate application response time metric."""
            import random
            return {
                "api_response_time": random.uniform(50, 500),  # ms
                "db_query_time": random.uniform(10, 100),      # ms
                "cache_hit_rate": random.uniform(0.7, 0.95)   # percentage
            }

        self.monitoring_tool.add_custom_metric(
            name="app_performance",
            metric_type=MetricType.GAUGE,
            description="Application performance metrics",
            collector=app_response_time_collector
        )

        # Business metrics
        def business_metrics_collector():
            """Simulate business metrics."""
            import random
            return {
                "active_users": random.randint(100, 1000),
                "transactions_per_minute": random.randint(10, 50),
                "revenue_per_hour": random.uniform(1000, 5000)
            }

        self.monitoring_tool.add_custom_metric(
            name="business_metrics",
            metric_type=MetricType.COUNTER,
            description="Key business metrics",
            collector=business_metrics_collector
        )

        # Security metrics
        def security_metrics_collector():
            """Simulate security metrics."""
            import random
            return {
                "failed_login_attempts": random.randint(0, 10),
                "suspicious_requests": random.randint(0, 5),
                "blocked_ips": random.randint(0, 20)
            }

        self.monitoring_tool.add_custom_metric(
            name="security_metrics",
            metric_type=MetricType.COUNTER,
            description="Security monitoring metrics",
            collector=security_metrics_collector
        )

        print("  ✅ Custom metrics configured")

    async def demonstrate_alert_workflow(self):
        """Demonstrate alert creation and management workflow."""
        print("🚨 Demonstrating alert workflow...")

        # Wait a bit for some metrics to be collected
        await asyncio.sleep(5)

        # Get current alerts
        active_alerts = len(self.monitoring_tool.active_alerts)
        print(f"  📊 Active alerts: {active_alerts}")

        # Simulate acknowledging alerts
        if self.monitoring_tool.active_alerts:
            alert_id = list(self.monitoring_tool.active_alerts.keys())[0]
            success = await self.monitoring_tool.acknowledge_alert(alert_id, "demo-user")
            if success:
                print(f"  ✅ Acknowledged alert: {alert_id}")

        # Get alert history
        history = await self.monitoring_tool.get_alert_history(limit=5)
        print(f"  📜 Recent alerts in history: {len(history)}")

    async def demonstrate_metrics_export(self):
        """Demonstrate various metrics export formats."""
        print("📤 Demonstrating metrics export...")

        try:
            # Export as JSON
            json_data = await self.monitoring_tool.export_metrics("json")
            json_file = "monitoring_export.json"
            with open(json_file, 'w') as f:
                f.write(json_data)
            print(f"  ✅ Exported JSON metrics to {json_file}")

            # Export as CSV
            csv_data = await self.monitoring_tool.export_metrics("csv")
            csv_file = "monitoring_export.csv"
            with open(csv_file, 'w') as f:
                f.write(csv_data)
            print(f"  ✅ Exported CSV metrics to {csv_file}")

            # Export Prometheus format (if available)
            prometheus_data = await self.monitoring_tool.export_metrics("prometheus")
            if prometheus_data and not prometheus_data.startswith("# Prometheus client not available"):
                prometheus_file = "monitoring_export.prom"
                with open(prometheus_file, 'w') as f:
                    f.write(prometheus_data)
                print(f"  ✅ Exported Prometheus metrics to {prometheus_file}")
            else:
                print("  ⚠️  Prometheus export not available (prometheus_client not installed)")

        except Exception as e:
            print(f"  ❌ Export error: {e}")

    async def display_real_time_dashboard(self, duration: int = 30):
        """Display a real-time dashboard for demonstration."""
        print(f"📺 Starting real-time dashboard (running for {duration} seconds)...")
        print("=" * 80)

        start_time = time.time()
        while time.time() - start_time < duration and self.demo_running:
            try:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')

                # Dashboard header
                print("🖥️  AGENTICAL MONITORING DASHBOARD")
                print("=" * 80)
                print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()

                # Get current metrics summary
                summary = await self.monitoring_tool.get_metrics_summary()

                # System metrics
                print("💻 SYSTEM METRICS")
                print("-" * 40)
                if 'system_metrics' in summary:
                    sys_metrics = summary['system_metrics']
                    print(f"  CPU Usage:    {sys_metrics.get('cpu_percent', 0):.1f}%")
                    print(f"  Memory Usage: {sys_metrics.get('memory_percent', 0):.1f}%")
                    print(f"  Disk Usage:   {sys_metrics.get('disk_usage', 0):.1f}%")
                print()

                # Target health
                print("🎯 TARGET HEALTH")
                print("-" * 40)
                if 'targets' in summary:
                    for name, target_info in summary['targets'].items():
                        status = target_info.get('status', 'unknown')
                        success_rate = target_info.get('success_rate', 0) * 100
                        avg_time = target_info.get('avg_response_time', 0)

                        status_icon = "✅" if status == "healthy" else "❌"
                        print(f"  {status_icon} {name}: {success_rate:.1f}% success, {avg_time:.0f}ms avg")
                print()

                # Alert summary
                print("🚨 ALERT SUMMARY")
                print("-" * 40)
                if 'alerts' in summary:
                    alerts = summary['alerts']
                    print(f"  Total Active: {alerts.get('total', 0)}")
                    print(f"  Unacknowledged: {alerts.get('unacknowledged', 0)}")

                    by_level = alerts.get('by_level', {})
                    if by_level:
                        for level, count in by_level.items():
                            level_icon = {
                                'info': 'ℹ️',
                                'warning': '⚠️',
                                'error': '❌',
                                'critical': '🔴',
                                'emergency': '🚨'
                            }.get(level, '📊')
                            print(f"  {level_icon} {level.title()}: {count}")
                print()

                # Health status
                health = self.monitoring_tool.get_health_status()
                status = health.get('status', 'unknown')
                status_icon = {
                    'healthy': '💚',
                    'warning': '💛',
                    'critical': '❤️',
                    'unknown': '🔘'
                }.get(status, '🔘')

                print("🏥 OVERALL HEALTH")
                print("-" * 40)
                print(f"  Status: {status_icon} {status.upper()}")
                print(f"  Uptime: {health.get('uptime', 0):.0f} seconds")
                print()

                print("=" * 80)
                print("Press Ctrl+C to stop the demo")

                # Wait before next update
                await asyncio.sleep(2)

            except KeyboardInterrupt:
                self.demo_running = False
                break
            except Exception as e:
                print(f"Dashboard error: {e}")
                await asyncio.sleep(2)

    async def run_demo(self):
        """Run the complete monitoring tool demonstration."""
        print("🚀 Starting Agentical Monitoring Tool Demo")
        print("=" * 50)

        self.demo_running = True

        try:
            # Step 1: Setup monitoring targets
            await self.setup_monitoring_targets()
            print()

            # Step 2: Setup custom metrics
            self.setup_custom_metrics()
            print()

            # Step 3: Start monitoring
            print("🔄 Starting monitoring system...")
            self.monitoring_tool.start_monitoring()
            print("  ✅ Monitoring system started")
            print()

            # Step 4: Let it run for a bit to collect some data
            print("⏳ Collecting initial metrics (10 seconds)...")
            await asyncio.sleep(10)
            print("  ✅ Initial metrics collected")
            print()

            # Step 5: Demonstrate alert workflow
            await self.demonstrate_alert_workflow()
            print()

            # Step 6: Show metrics summary
            print("📊 Current metrics summary:")
            summary = await self.monitoring_tool.get_metrics_summary()
            print(json.dumps(summary, indent=2, default=str)[:500] + "...")
            print()

            # Step 7: Demonstrate exports
            await self.demonstrate_metrics_export()
            print()

            # Step 8: Real-time dashboard
            await self.display_real_time_dashboard(30)

        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            await self.monitoring_tool.stop_monitoring()
            await self.monitoring_tool.cleanup()
            print("  ✅ Cleanup completed")

            print("\n📁 Generated files:")
            for filename in ["monitoring_export.json", "monitoring_export.csv", "monitoring_export.prom"]:
                if os.path.exists(filename):
                    size = os.path.getsize(filename)
                    print(f"  📄 {filename} ({size} bytes)")

            print("\n✨ Demo completed successfully!")


async def main():
    """Main demo function."""
    print("Welcome to the Agentical Monitoring Tool Demo!")
    print()
    print("This demo will showcase:")
    print("  • System monitoring capabilities")
    print("  • Target health checking")
    print("  • Custom metrics collection")
    print("  • Alert management")
    print("  • Data export in multiple formats")
    print("  • Real-time dashboard")
    print()

    # Ask user if they want to continue
    try:
        response = input("Continue with the demo? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Demo cancelled.")
            return
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return

    print()

    # Run the demo
    demo = MonitoringDemo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    except Exception as e:
        print(f"Demo failed: {e}")
