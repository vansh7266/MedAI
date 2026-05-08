# GCP Deployment Guide

## Prerequisites

- Google Cloud account (free tier)
- GitHub repo connected

## Step 1: Enable APIs

- Cloud Run API
- Artifact Registry API
- Cloud Build API
- Vertex AI API

## Step 2: Create Service Account

- Name: medai-deployer
- Roles: Cloud Run Admin, Storage Admin, Artifact Registry Writer, Vertex AI User
- Download JSON key

## Step 3: GitHub Secrets

Add to Settings > Secrets:

- GCP_PROJECT_ID: your-project-id
- GCP_SA_KEY: paste entire JSON key
- GCP_REGION: us-central1

## Step 4: Push to main branch

- GitHub Actions will auto-deploy

## Step 5: Verify

- Check Cloud Run console for service URL
- Test /health endpoint

## Cost Estimates (Free Tier)

- Cloud Run: 2M requests/month free
- Artifact Registry: 0.5GB storage free
- Vertex AI: ~$0.0001 per 1K tokens for Gemini Flash Lite
