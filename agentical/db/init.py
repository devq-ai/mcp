"""
Database Initialization Script

This script initializes the database schema, creates tables, and sets up initial data.
It uses SQLAlchemy ORM and follows the DevQ.ai best practices for database setup.

Features:
- Database schema creation and versioning
- Index creation for query optimization
- Initial data seeding
- Database migration support
- Integration with Logfire for observability
"""

import os
import logging
import argparse
import asyncio
from typing import List, Dict, Any, Optional

import logfire
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from agentical.db import Base, engine, async_engine, initialize_database, initialize_async_database
from agentical.db.models.user import User, Role

# Configure logging
logger = logging.getLogger(__name__)

# Default admin user credentials for initial setup
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_EMAIL = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "changeme")

# Default roles
DEFAULT_ROLES = [
    {
        "name": "admin",
        "description": "Administrator with full permissions",
        "permissions": ["admin", "read", "write", "delete"]
    },
    {
        "name": "user",
        "description": "Regular user with basic permissions",
        "permissions": ["read"]
    },
    {
        "name": "editor",
        "description": "Editor with read and write permissions",
        "permissions": ["read", "write"]
    },
    {
        "name": "agent",
        "description": "Agent with specific permissions",
        "permissions": ["read", "agent:execute"]
    }
]


def create_indices(engine) -> None:
    """
    Create database indices for query optimization.
    
    Args:
        engine: SQLAlchemy engine
    """
    with logfire.span("Create database indices"):
        try:
            # Connect to database
            conn = engine.connect()
            
            # Get inspector for checking existing indices
            inspector = inspect(engine)
            
            # Create indices for User model
            user_indices = {
                "ix_user_email": "CREATE INDEX IF NOT EXISTS ix_user_email ON user (email)",
                "ix_user_username": "CREATE INDEX IF NOT EXISTS ix_user_username ON user (username)",
                "ix_user_created_at": "CREATE INDEX IF NOT EXISTS ix_user_created_at ON user (created_at)",
                "ix_user_is_active": "CREATE INDEX IF NOT EXISTS ix_user_is_active ON user (is_active)"
            }
            
            # Check existing indices on user table
            existing_indices = inspector.get_indexes("user")
            existing_index_names = [idx["name"] for idx in existing_indices]
            
            # Create missing indices
            for name, sql in user_indices.items():
                if name not in existing_index_names:
                    conn.execute(text(sql))
                    logger.info(f"Created index: {name}")
                    
            # Create indices for other tables as needed
            # ...
            
            # Commit changes
            conn.commit()
            logger.info("Database indices created successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating indices: {e}")
            logfire.error(
                "Database index creation error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        finally:
            if 'conn' in locals():
                conn.close()


def create_initial_roles(session) -> Dict[str, Role]:
    """
    Create initial roles in the database.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary mapping role names to Role objects
    """
    with logfire.span("Create initial roles"):
        try:
            # Check if roles already exist
            existing_roles = session.query(Role).all()
            if existing_roles:
                logger.info(f"Found {len(existing_roles)} existing roles")
                return {role.name: role for role in existing_roles}
            
            # Create roles from defaults
            roles = {}
            for role_data in DEFAULT_ROLES:
                import json
                role = Role(
                    name=role_data["name"],
                    description=role_data["description"],
                    permissions=json.dumps(role_data["permissions"])
                )
                session.add(role)
                roles[role.name] = role
                logger.info(f"Created role: {role.name}")
            
            # Commit changes
            session.commit()
            logger.info(f"Created {len(roles)} initial roles")
            return roles
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating roles: {e}")
            logfire.error(
                "Database role creation error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise


def create_admin_user(session, roles: Dict[str, Role]) -> Optional[User]:
    """
    Create admin user if it doesn't exist.
    
    Args:
        session: Database session
        roles: Dictionary mapping role names to Role objects
        
    Returns:
        Created admin user or None if it already exists
    """
    with logfire.span("Create admin user"):
        try:
            # Check if admin user already exists
            admin = session.query(User).filter(
                User.username == DEFAULT_ADMIN_USERNAME
            ).first()
            
            if admin:
                logger.info(f"Admin user '{DEFAULT_ADMIN_USERNAME}' already exists")
                return None
            
            # Create admin user
            admin = User(
                username=DEFAULT_ADMIN_USERNAME,
                email=DEFAULT_ADMIN_EMAIL,
                is_verified=True,
                is_active=True
            )
            admin.password = DEFAULT_ADMIN_PASSWORD
            
            # Assign admin role
            if "admin" in roles:
                admin.roles.append(roles["admin"])
            
            session.add(admin)
            session.commit()
            logger.info(f"Created admin user: {admin.username}")
            
            # Log warning to change default password
            logger.warning("Default admin user created. Please change the password immediately!")
            return admin
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating admin user: {e}")
            logfire.error(
                "Database admin user creation error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise


def initialize_sync() -> None:
    """Initialize database synchronously."""
    with logfire.span("Initialize database"):
        # Create tables
        initialize_database(drop_all=False)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Create indices
            create_indices(engine)
            
            # Create initial data
            roles = create_initial_roles(session)
            create_admin_user(session, roles)
            
            logger.info("Database initialization completed successfully")
        finally:
            session.close()


async def initialize_async() -> None:
    """Initialize database asynchronously."""
    with logfire.span("Initialize database async"):
        # Create tables
        await initialize_async_database(drop_all=False)
        
        # Create session
        async with AsyncSession(async_engine) as session:
            # Create indices
            create_indices(async_engine)
            
            # Create initial data
            roles = create_initial_roles(session)
            create_admin_user(session, roles)
            
            logger.info("Async database initialization completed successfully")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Initialize database")
    parser.add_argument("--async-mode", action="store_true", help="Use async initialization")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure Logfire
    try:
        logfire.configure(
            token=os.getenv("LOGFIRE_TOKEN"),
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "agentical-db-init"),
            environment=os.getenv("ENVIRONMENT", "development")
        )
    except Exception as e:
        logger.warning(f"Failed to configure Logfire: {e}")
    
    try:
        if args.async_mode:
            # Run async initialization
            asyncio.run(initialize_async())
        else:
            # Run sync initialization
            initialize_sync()
            
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logfire.error(
            "Database initialization failed",
            error=str(e),
            error_type=type(e).__name__
        )
        exit(1)


if __name__ == "__main__":
    main()