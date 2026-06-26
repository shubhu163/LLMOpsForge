# API Documentation

_Last updated: 2025-02-10_

## Base URL

All API requests are made to:

```
https://api.acmecloud.example/v1
```

## Authentication

The API uses **Bearer token** authentication. Include your API key in the
`Authorization` header:

```
Authorization: Bearer <YOUR_API_KEY>
```

API keys are created in the dashboard under **Settings → API Keys**. Keys can be
scoped to read-only or read-write.

## Rate Limits

- Pro plan: **100 requests per second** per API key.
- Starter plan: **10 requests per second** per API key.
- Exceeding the limit returns HTTP status **429 Too Many Requests** with a
  `Retry-After` header indicating how many seconds to wait.

## Pagination

List endpoints are paginated using **cursor-based** pagination. Responses
include a `next_cursor` field. Pass it as the `cursor` query parameter to fetch
the next page. The default page size is **50** and the maximum is **200**.

## Errors

Errors return a JSON body with the shape:

```json
{ "error": { "code": "invalid_request", "message": "..." } }
```

Common status codes:

- **400** invalid request
- **401** missing or invalid API key
- **403** insufficient permissions
- **404** resource not found
- **429** rate limited

## Versioning

The API is versioned in the URL path (`/v1`). Breaking changes are released
under a new version. Deprecated versions are supported for **12 months** after a
new version is released.

## Webhooks

You can register webhook endpoints to receive event notifications. Webhook
payloads are signed with an HMAC-SHA256 signature in the `X-Acme-Signature`
header so you can verify authenticity. Failed webhook deliveries are retried
with exponential backoff for up to **24 hours**.

## SDKs

Official SDKs are available for **Python** and **JavaScript/TypeScript**. Both
are published under the `@acmecloud` namespace and follow semantic versioning.
