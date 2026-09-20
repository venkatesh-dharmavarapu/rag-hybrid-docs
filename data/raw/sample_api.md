# Internal Payment Gateway API

## Authentication
All API requests must include the header `Authorization: Bearer <token>`.
Tokens expire after 15 minutes of inactivity.

## Error Codes
- `ERR_AUTH_001`: Token missing or malformed.
- `ERR_RATE_LIMIT`: Exceeded 500 requests per minute.
- `ERR_PAYMENT_FAILED`: Downstream processor rejected transaction.