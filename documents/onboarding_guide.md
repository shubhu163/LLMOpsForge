# Onboarding Guide

_Last updated: 2025-02-12_

## Welcome

This guide walks new customers through setting up their Acme Cloud account, from
sign-up to their first successful API call.

## Step 1: Create an Account

Sign up at `app.acmecloud.example/signup` using a work email. New accounts start
on the **Starter** (free) plan. You can upgrade at any time from
**Settings → Billing**.

## Step 2: Verify Your Email

After signing up, you will receive a verification email. Click the link to
activate your account. Verification links expire after **24 hours**. If your link
expires, you can request a new one from the login screen.

## Step 3: Create Your First Project

From the dashboard, click **New Project**, give it a name, and choose a region.
Each project gets an isolated set of resources and its own API keys.

## Step 4: Generate an API Key

Open **Settings → API Keys** and click **Create Key**. Copy the key immediately —
for security, the full key is shown **only once** and cannot be retrieved later.
If you lose a key, revoke it and create a new one.

## Step 5: Make Your First API Call

Use the quickstart snippet to confirm connectivity:

```bash
curl https://api.acmecloud.example/v1/ping \
  -H "Authorization: Bearer <YOUR_API_KEY>"
```

A successful response returns `{ "status": "ok" }`.

## Inviting Teammates

Account owners can invite teammates from **Settings → Members**. Invited users
receive an email invitation that expires after **7 days**. Available roles are
**Owner**, **Admin**, **Member**, and **Viewer**.

## Roles and Permissions

- **Owner**: full access including billing and account deletion.
- **Admin**: manage projects, members, and API keys; no billing access.
- **Member**: create and edit resources within assigned projects.
- **Viewer**: read-only access.

## Getting Help

- In-app help: click the **?** icon in the bottom-right.
- Documentation: `docs.acmecloud.example`.
- Support email: `support@acmecloud.example`.

Pro and Enterprise customers receive prioritized support response times as
described in the Pricing document.
