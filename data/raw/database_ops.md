# Database Operations and Rate Limiting

## Rate Limiting Policy
All internal services are throttled to a maximum of 500 requests per minute per IP address.
When an application exceeds this threshold, the API responds with `ERR_RATE_LIMIT`.

## Database Failover
PostgreSQL replication operates in hot-standby mode.
Failover to the standby cluster initiates automatically within 30 seconds of primary heartbeat loss.
Contact DevOps at `devops-alerts@internal.company` if automatic failover fails.