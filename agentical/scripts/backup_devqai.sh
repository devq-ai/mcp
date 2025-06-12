#!/bin/bash

# Configuration
SOURCE_DIR="/Users/dionedge/devqai"
BACKUP_DIR="/Users/dionedge/backups"
BACKUP_PREFIX="devqai_backup"
MAX_BACKUPS=7
EMAIL="dion@devq.ai"
LOG_FILE="${BACKUP_DIR}/backup.log"

# Function to send email
send_email() {
    local subject="$1"
    local body="$2"
    echo "Subject: $subject" > /tmp/email.txt
    echo "" >> /tmp/email.txt
    echo "$body" >> /tmp/email.txt
    sendmail "$EMAIL" < /tmp/email.txt
    rm /tmp/email.txt
}

# Function to log messages
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Get current date
DATE=$(date +%Y%m%d)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# Create backup filename
BACKUP_FILE="${BACKUP_DIR}/${BACKUP_PREFIX}_${DATE}.tar.gz"

# Initialize email body
EMAIL_BODY="Backup Report for ${DATE}\n\n"
EMAIL_BODY+="Start Time: ${START_TIME}\n"

# Create backup
log_message "Creating backup of ${SOURCE_DIR}..."
tar -czf "$BACKUP_FILE" \
    --exclude="node_modules" \
    --exclude=".git" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".DS_Store" \
    --exclude="*.log" \
    --exclude="venv" \
    -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")" 2>&1 | tee -a "$LOG_FILE"

# Check if backup was successful
if [ $? -eq 0 ]; then
    SUCCESS=true
    log_message "Backup created successfully: $BACKUP_FILE"
    EMAIL_BODY+="Status: ✅ Success\n"
else
    SUCCESS=false
    log_message "Error creating backup"
    EMAIL_BODY+="Status: ❌ Failed\n"
fi

# Remove old backups (keep only last MAX_BACKUPS days)
log_message "Cleaning up old backups..."
OLD_BACKUPS=$(ls -t "${BACKUP_DIR}/${BACKUP_PREFIX}"_*.tar.gz 2>/dev/null | tail -n +$((MAX_BACKUPS + 1)))
if [ ! -z "$OLD_BACKUPS" ]; then
    echo "$OLD_BACKUPS" | xargs rm
    log_message "Removed old backups: $OLD_BACKUPS"
    EMAIL_BODY+="Old backups removed: Yes\n"
else
    log_message "No old backups to remove"
    EMAIL_BODY+="Old backups removed: No\n"
fi

# Get backup details
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
BACKUP_SIZE=$(du -h "$BACKUP_FILE" 2>/dev/null | cut -f1)
TOTAL_BACKUPS=$(ls "${BACKUP_DIR}/${BACKUP_PREFIX}"_*.tar.gz 2>/dev/null | wc -l)

# Add details to email body
EMAIL_BODY+="End Time: ${END_TIME}\n"
EMAIL_BODY+="Backup Location: ${BACKUP_FILE}\n"
EMAIL_BODY+="Backup Size: ${BACKUP_SIZE}\n"
EMAIL_BODY+="Total Backups: ${TOTAL_BACKUPS}\n\n"

# Add list of current backups
EMAIL_BODY+="Current Backups:\n"
ls -lh "${BACKUP_DIR}/${BACKUP_PREFIX}"_*.tar.gz 2>/dev/null | \
    awk '{print $9 " (" $5 ")"}' | \
    while read line; do
        EMAIL_BODY+="$line\n"
    done

# Send email notification
if [ "$SUCCESS" = true ]; then
    send_email "✅ DevQAI Backup Success - ${DATE}" "$EMAIL_BODY"
else
    send_email "❌ DevQAI Backup Failed - ${DATE}" "$EMAIL_BODY"
fi

# Final log entry
log_message "Backup process completed"
log_message "Email notification sent to $EMAIL"

# Display final status
echo "----------------------------------------"
echo "Backup Summary:"
echo "Status: $([ "$SUCCESS" = true ] && echo "Success ✅" || echo "Failed ❌")"
echo "Location: $BACKUP_FILE"
echo "Size: $BACKUP_SIZE"
echo "Email sent to: $EMAIL"
echo "Log file: $LOG_FILE"
echo "----------------------------------------"