# Product Security

_Last updated: 2025-02-01_

## Data Encryption

All customer data is encrypted **in transit** using TLS 1.2 or higher, and
**at rest** using AES-256. Encryption keys are managed through a dedicated key
management service and rotated **every 90 days**.

## Authentication

- Password-based login requires a minimum of **12 characters**.
- **Multi-factor authentication (MFA)** is available on all plans and is
  **required** for Enterprise accounts.
- Single Sign-On (SSO) via SAML 2.0 is available on the Enterprise plan.

## Compliance

Acme Cloud is **SOC 2 Type II** certified and undergoes an independent audit
annually. We are also **GDPR** compliant. A copy of the current SOC 2 report is
available under NDA upon request.

## Data Residency

By default, customer data is stored in the **United States (us-east)**.
Enterprise customers may request data residency in the **European Union
(eu-central)** region.

## Backups

Customer data is backed up **daily**, and backups are retained for **35 days**.
Backups are encrypted with the same AES-256 standard as primary storage.

## Incident Response

Acme maintains a documented incident response plan. Confirmed security incidents
affecting customer data are disclosed to affected customers within **72 hours**
of confirmation.

## Vulnerability Disclosure

Security researchers may report vulnerabilities to `security@acmecloud.example`.
Acme operates a responsible disclosure program and aims to acknowledge reports
within **2 business days**.

## Access Controls

Internal access to production systems follows the principle of **least
privilege** and requires MFA. All production access is logged and reviewed
quarterly.
