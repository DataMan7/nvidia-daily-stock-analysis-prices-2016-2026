# 🚀 NVIDIA Stock Prediction - Deployment Guide

This guide covers deployment options for the NVIDIA stock prediction system on various platforms.

## 📋 System Overview

- **FastAPI Service**: REST API for real-time predictions (`scripts/api.py`)
- **Streamlit Dashboard**: Interactive monitoring interface (`scripts/dashboard.py`)
- **MLflow Tracking**: Experiment management and model registry
- **Training Scripts**: Automated model training and evaluation

## 🐳 Local Deployment (Docker)

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Access services:
# - API: http://localhost:8000/docs (FastAPI)
# - Dashboard: http://localhost:8501 (Streamlit)
# - MLflow: http://localhost:5000 (Tracking UI)
```

### Individual Services

```bash
# API only
docker-compose up api

# Dashboard only
docker-compose up dashboard

# Development environment
docker-compose --profile dev up dev
```

### Docker Commands

```bash
# Build production image
docker build --target production -t nvidia-stock-prediction .

# Run API
docker run -p 8000:8000 -v $(pwd)/models:/app/models nvidia-stock-prediction

# Run dashboard
docker run -p 8501:8501 -v $(pwd)/models:/app/models nvidia-stock-prediction \
  streamlit run scripts/dashboard.py --server.port=8501 --server.address=0.0.0.0
```

## ☁️ Cloud Deployment Platforms

### 1. **Railway** (Recommended for Quick MVP)
**Best for**: Fast prototyping, automatic deployments

**Pros**: Free tier, GitHub integration, automatic SSL
**Cons**: Limited customization, vendor lock-in

**Setup**:
1. Connect GitHub repository
2. Railway auto-detects Python app
3. Set environment variables
4. Deploy automatically on git push

**Cost**: Free tier available, $5-10/month for basic usage

### 2. **Render** (Great for APIs)
**Best for**: Web services, APIs, background jobs

**Pros**: Free tier, managed databases, auto-scaling
**Cons**: Cold starts, limited free hours

**Setup**:
```yaml
# render.yaml
services:
  - type: web
    name: nvidia-stock-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/api.py
  - type: web
    name: nvidia-stock-dashboard
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run scripts/dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

**Cost**: 750 free hours/month, $7/month after

### 3. **Fly.io** (Excellent for Global Apps)
**Best for**: Global distribution, low latency

**Pros**: Global edge network, generous free tier
**Cons**: Complex setup, learning curve

**Setup**:
```toml
# fly.toml
app = "nvidia-stock-prediction"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000
  force_https = true

[[mounts]]
  source = "models"
  destination = "/app/models"
```

**Cost**: 3 free VMs, $10-20/month for global deployment

### 4. **Google Cloud Run** (Enterprise-Grade)
**Best for**: Scalable production, enterprise features

**Pros**: Auto-scaling, managed infrastructure, enterprise security
**Cons**: Complex setup, higher cost

**Setup**:
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/nvidia-stock', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/nvidia-stock']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'nvidia-stock-api'
      - '--image=gcr.io/$PROJECT_ID/nvidia-stock'
      - '--platform=managed'
      - '--port=8000'
```

**Cost**: Pay-per-use, ~$0.10 per 1000 requests

### 5. **AWS Fargate** (Cloud-Native)
**Best for**: AWS ecosystem, advanced networking

**Pros**: Deep AWS integration, advanced features
**Cons**: Complex setup, AWS learning curve

**Setup**: Use AWS Copilot CLI
```bash
# Install AWS Copilot
brew install aws/tap/copilot-cli

# Deploy
copilot init
copilot deploy
```

**Cost**: ~$15-30/month for basic setup

### 6. **DigitalOcean App Platform** (Simple & Affordable)
**Best for**: Small teams, straightforward deployment

**Pros**: Simple UI, good documentation, affordable
**Cons**: Limited advanced features

**Setup**: Connect GitHub repo, auto-deploy on push

**Cost**: $12/month for basic app

### 7. **Vercel** (Frontend Focus)
**Best for**: Dashboard-only deployment

**Pros**: Excellent for Streamlit, global CDN
**Cons**: API deployment more complex

**Setup**:
```json
// vercel.json
{
  "version": 2,
  "builds": [
    {
      "src": "scripts/dashboard.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "scripts/dashboard.py"
    }
  ]
}
```

**Cost**: Generous free tier, $20-40/month

## 🏢 Enterprise Platforms

### **Azure ML** (Microsoft Ecosystem)
- **Best for**: Enterprise Microsoft shops
- **Cost**: Pay-per-use, ~$10-50/month
- **Features**: MLOps integration, Azure AI integration

### **SageMaker** (AWS ML Platform)
- **Best for**: Large-scale ML operations
- **Cost**: $10-100/month depending on usage
- **Features**: Complete MLOps pipeline, model monitoring

### **Vertex AI** (Google Cloud)
- **Best for**: Google Cloud users
- **Cost**: Pay-per-use, ~$20-200/month
- **Features**: AutoML, model monitoring, explainability

## 📊 Platform Comparison

| Platform | Free Tier | Ease of Use | Scaling | Cost/Month | Best For |
|----------|-----------|-------------|---------|------------|----------|
| Railway | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $0-10 | Quick MVPs |
| Render | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $0-15 | APIs & Dashboards |
| Fly.io | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $0-20 | Global apps |
| Cloud Run | ❌ | ⭐⭐ | ⭐⭐⭐⭐⭐ | $0-50 | Enterprise |
| DigitalOcean | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐ | $12+ | Small teams |
| Vercel | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $0-40 | Dashboards |

## 🚀 Recommended Deployment Strategy

### **Phase 1: Development & Testing**
```bash
# Local Docker development
docker-compose --profile dev up dev
```

### **Phase 2: MVP Launch (Recommended: Railway)**
1. **Railway** for API ($0/month free)
2. **Vercel** for dashboard ($0/month free)
3. **GitHub** for code hosting

### **Phase 3: Production Scale**
- **Render** for balanced cost/performance
- **Fly.io** for global distribution
- **Cloud Run** for enterprise features

## 🔧 Environment Variables

### Required for Production
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# MLflow (optional)
MLFLOW_TRACKING_URI=http://mlflow:5000

# Model paths
MODEL_DIR=/app/models
DATA_DIR=/app/data
```

### Platform-Specific Setup

#### Railway
```bash
# railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python scripts/api.py"
  }
}
```

#### Render
```yaml
# render.yaml
services:
  - type: web
    name: nvidia-stock-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python scripts/api.py"
```

## 📈 Monitoring & Maintenance

### Health Checks
- API: `GET /health` - System status
- Dashboard: Built-in Streamlit metrics
- MLflow: Experiment tracking UI

### Logging
```python
# Structured logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Backup Strategy
- **Models**: Version in MLflow registry
- **Data**: Regular snapshots
- **Code**: Git versioning
- **Metrics**: Export to cloud storage

## 🔒 Security Considerations

### API Security
- **CORS**: Configure allowed origins
- **Rate Limiting**: Implement request limits
- **Authentication**: Add API keys for production
- **HTTPS**: Always use SSL in production

### Data Security
- **Encryption**: Encrypt sensitive data
- **Access Control**: Limit data access
- **Audit Logs**: Track all access

## 💡 Pro Tips

1. **Start Simple**: Railway/Render for first deployment
2. **Monitor Costs**: Set up billing alerts
3. **Automate**: Use GitHub Actions for CI/CD
4. **Scale Gradually**: Start with one service, add others
5. **Backup Everything**: Models, data, configurations
6. **Test Locally**: Always test Docker setup locally first

## 📞 Support & Resources

- **Railway Docs**: https://docs.railway.app/
- **Render Docs**: https://docs.render.com/
- **Fly.io Docs**: https://fly.io/docs/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Streamlit Docs**: https://docs.streamlit.io/

---

**Ready to deploy? Start with Railway for the fastest path to production! 🚀**