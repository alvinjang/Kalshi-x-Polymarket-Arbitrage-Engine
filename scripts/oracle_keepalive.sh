#!/bin/bash
# oracle_keepalive.sh -- CPU keep-alive to prevent Oracle free-tier VM reclamation.
#
# Oracle reclaims idle free-tier VMs after ~7 days of low CPU usage.
# This script generates a brief CPU burst every 10 minutes (via cron)
# to keep the VM alive.
#
# Cron entry (installed by setup_oracle.sh):
#   */10 * * * * bash ~/DS340-Project/scripts/oracle_keepalive.sh >> ~/DS340-Project/logs/keepalive.log 2>&1

# Brief CPU burst: hash 10MB of random data
dd if=/dev/urandom bs=1M count=10 2>/dev/null | md5sum > /dev/null 2>&1

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): keepalive ping"
