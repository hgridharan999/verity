# Verity - Content Verification Engine

Enterprise-grade content authenticity verification engine that determines whether uploaded images and videos are authentic or AI-generated through a multi-layered verification pipeline.

## Features

### 🔐 Multi-Layered Verification Pipeline

1. **C2PA Signature Verification** - Cryptographic proof of authenticity
2. **Hardware Authentication** - Device signature and attestation checking
3. **Metadata Analysis** - EXIF/XMP extraction and consistency validation
4. **Contextual Verification** - Creator reputation and cross-referencing
5. **ML Detection** - AI-based detection using ensemble models

### 🎯 Key Capabilities

- **Verification-First Architecture**: Prioritizes cryptographic proof over pattern matching
- **Future-Proof Technology**: Won't become obsolete as AI generators improve
- **Multi-Format Support**: Images (JPEG, PNG, WebP, TIFF, HEIC) and Videos (MP4, MOV, AVI, MKV, WebM)
- **Enterprise Ready**: Scalable, secure, and compliant with industry standards
- **Vertical Agnostic**: Adaptable for insurance, legal, e-commerce, and journalism use cases

### 📊 Verification Outputs

- **Trust Score** (0-100): Overall authenticity confidence
- **Risk Category**: verified | authentic_high_confidence | likely_authentic | uncertain | likely_synthetic | synthetic_high_confidence | fraudulent
- **Evidence Trail**: Complete audit trail of verification steps
- **Detailed Reports**: Stage-by-stage breakdown with findings

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL 15+ (if running without Docker)
- Redis 7+ (if running without Docker)

### Running with Docker (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd verity
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start all services**
```bash
docker-compose up -d
```

4. **Initialize the database**
```bash
docker-compose exec app alembic upgrade head
```

5. **Access the API**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

### Running Locally

1. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set up PostgreSQL and Redis**
```bash
# Start PostgreSQL
# Start Redis
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your database and Redis URLs
```

4. **Run database migrations**
```bash
alembic upgrade head
```

5. **Start the application**
```bash
uvicorn app.main:app --reload
```

## Usage

### API Endpoints

#### Verify Content

```bash
POST /api/v1/verify
Content-Type: multipart/form-data

Parameters:
- file: File to verify (required)
- priority: standard|expedited (default: standard)
- include_detailed_report: true|false (default: true)
- vertical: insurance|legal|ecommerce|news (optional)
- force_full_pipeline: true|false (default: false)
```

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "X-API-Key: your-api-key" \
  -F "file=@/path/to/image.jpg" \
  -F "priority=standard" \
  -F "include_detailed_report=true"
```

**Example using Python:**

```python
import requests

url = "http://localhost:8000/api/v1/verify"
headers = {"X-API-Key": "your-api-key"}
files = {"file": open("image.jpg", "rb")}
data = {
    "priority": "standard",
    "include_detailed_report": True,
    "vertical": "insurance"
}

response = requests.post(url, headers=headers, files=files, data=data)
result = response.json()

print(f"Trust Score: {result['trust_score']}")
print(f"Risk Category: {result['risk_category']}")
print(f"Recommendation: {result['recommendation']}")
```

**Response:**

```json
{
  "verification_id": "ver_abc123def456",
  "timestamp": "2025-12-02T15:30:00Z",
  "file_hash": "sha256:abc123...",
  "trust_score": 78.5,
  "confidence": 85.0,
  "risk_category": "likely_authentic",
  "processing_time_ms": 2847,
  "verification_stages": [
    {
      "stage": 1,
      "name": "C2PA Verification",
      "status": "not_present",
      "duration_ms": 245,
      "contribution": 0
    },
    ...
  ],
  "key_findings": [
    "Valid hardware signature from Canon EOS R5",
    "Consistent metadata across all fields",
    "ML models indicate likely authentic"
  ],
  "risk_factors": [],
  "recommendation": "Content appears authentic with high confidence..."
}
```

#### Get Verification Result

```bash
GET /api/v1/verify/{verification_id}
```

#### Delete Verification (GDPR Compliance)

```bash
DELETE /api/v1/verify/{verification_id}
```

#### Health Check

```bash
GET /api/v1/health
```

## Architecture

### System Components

```
┌─────────────────────────────────────────┐
│         API Gateway Layer               │
│  - FastAPI with async support           │
│  - Authentication & Rate Limiting       │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│      Content Ingestion Module           │
│  - File validation & malware scanning   │
│  - Metadata extraction                  │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   5-Stage Verification Pipeline         │
│  1. C2PA Signature Verification         │
│  2. Hardware Authentication             │
│  3. Metadata Analysis                   │
│  4. Contextual Verification             │
│  5. ML Detection                        │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│    Aggregation & Scoring Engine         │
│  - Trust score calculation              │
│  - Evidence compilation                 │
└─────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: Python 3.11+, FastAPI
- **Database**: PostgreSQL 15+ with async support
- **Cache/Queue**: Redis
- **ML Framework**: PyTorch, OpenCV, scikit-image
- **Container**: Docker, Docker Compose
- **Logging**: structlog
- **Monitoring**: Prometheus-compatible metrics

## Configuration

Key environment variables (see `.env.example` for complete list):

```bash
# Application
APP_NAME=Verity
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/verity

# Redis
REDIS_URL=redis://localhost:6379/0

# File Upload
MAX_UPLOAD_SIZE_MB=500
ALLOWED_IMAGE_FORMATS=jpg,jpeg,png,webp,tiff,heic
ALLOWED_VIDEO_FORMATS=mp4,mov,avi,mkv,webm

# ML Models
ML_DEVICE=cuda  # or cpu
ML_MODEL_DIR=./models

# Security
SECRET_KEY=your-secret-key-here
API_KEY_EXPIRY_DAYS=365

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

## Development

### Project Structure

```
verity/
├── app/
│   ├── api/                    # API endpoints
│   │   └── v1/
│   │       └── endpoints/
│   ├── core/                   # Core utilities
│   │   ├── exceptions.py
│   │   └── security.py
│   ├── db/                     # Database configuration
│   ├── models/                 # Database models & schemas
│   ├── services/              # Business logic
│   │   ├── verification/      # Verification stages
│   │   ├── ingestion.py
│   │   ├── scoring_engine.py
│   │   └── verification_pipeline.py
│   ├── utils/                 # Utilities
│   ├── config.py              # Configuration
│   └── main.py                # Application entry point
├── tests/                     # Test suite
├── docker-compose.yml         # Docker services
├── Dockerfile                 # Application container
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Performance

### Target Metrics

- **Processing Speed**: <3 seconds for images, <30 seconds for videos (p95)
- **Verification Accuracy**: 95%+ on signed content, 75%+ on unsigned content
- **API Uptime**: 99.9% availability
- **Throughput**: 1M+ verifications/day with proper scaling

### Optimization

- Early exit on C2PA verified content (~1 second)
- Parallel stage execution where possible
- GPU acceleration for ML inference
- Caching of expensive computations
- Database query optimization

## Security

### Features

- **End-to-end encryption** (TLS 1.3)
- **API key authentication**
- **Rate limiting** per key
- **File malware scanning**
- **Secure file storage** with automatic cleanup
- **GDPR compliant** data deletion
- **Audit logging** of all operations

### Best Practices

1. Always use HTTPS in production
2. Rotate API keys regularly
3. Set up rate limiting appropriate for your use case
4. Configure file retention policies
5. Enable monitoring and alerting
6. Regular security audits

## Deployment

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set strong `SECRET_KEY`
- [ ] Configure proper `DATABASE_URL`
- [ ] Set up SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Set up backup strategy
- [ ] Configure auto-scaling if needed
- [ ] Test disaster recovery procedures

### Kubernetes Deployment

```yaml
# Example Kubernetes deployment (simplified)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: verity
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: verity
        image: verity:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: verity-secrets
              key: database-url
```

## Troubleshooting

### Common Issues

**Database connection errors:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres
```

**Redis connection errors:**
```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
redis-cli ping
```

**Out of memory during ML inference:**
```bash
# Reduce batch size in .env
ML_BATCH_SIZE=4

# Or use CPU instead of GPU
ML_DEVICE=cpu
```

## Roadmap

### Phase 1 (Current)
- ✅ Core verification pipeline
- ✅ C2PA, metadata, and ML detection
- ✅ RESTful API
- ✅ Docker deployment

### Phase 2
- [ ] Advanced ML models (transformer-based)
- [ ] Batch verification
- [ ] Webhook support
- [ ] Video verification optimization

### Phase 3
- [ ] Web dashboard
- [ ] Browser extension
- [ ] Mobile SDKs
- [ ] Real-time verification streaming

### Phase 4
- [ ] Blockchain integration
- [ ] Decentralized verification
- [ ] Industry-specific models
- [ ] Custom model training platform

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license here]

## Support

- Documentation: http://localhost:8000/docs
- Issues: [GitHub Issues]
- Email: support@verity.example.com

## Acknowledgments

- C2PA Consortium for content authenticity standards
- Open source ML detection research community
- FastAPI and Python ecosystem

---

**Built with ❤️ for content authenticity**
