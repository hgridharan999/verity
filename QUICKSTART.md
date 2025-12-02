# Verity Quick Start Guide

Get up and running with Verity in minutes!

## Option 1: Docker (Recommended)

### Prerequisites
- Docker Desktop installed and running
- 4GB+ RAM available

### Steps

1. **Clone and navigate to the project**
```bash
cd verity
```

2. **Start all services**
```bash
docker-compose up -d
```

This starts:
- PostgreSQL database
- Redis cache/queue
- Verity API server
- Celery worker (background tasks)

3. **Initialize the database**
```bash
docker-compose exec app alembic upgrade head
```

4. **Verify it's working**
```bash
curl http://localhost:8000/api/v1/health
```

5. **Access the API documentation**
Open http://localhost:8000/docs in your browser

### Test the API

```bash
# Create a test image (or use your own)
curl -X POST "http://localhost:8000/api/v1/verify" \
  -F "file=@test_image.jpg" \
  -F "priority=standard" \
  -F "include_detailed_report=true"
```

## Option 2: Local Development

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+

### Steps

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your database and Redis URLs
```

4. **Initialize database**
```bash
python scripts/init_db.py
alembic upgrade head
```

5. **Run the application**
```bash
uvicorn app.main:app --reload
```

6. **Access the API**
http://localhost:8000/docs

## Using the Makefile

For convenience, you can use the Makefile:

```bash
# Install dependencies
make dev-install

# Initialize everything
make init

# Run migrations
make migrate

# Start the app
make run

# With Docker
make docker-up
make docker-setup  # First time only
```

## Testing

```bash
# Run tests
make test

# With coverage
make test-cov

# View coverage report
open htmlcov/index.html
```

## Example Verification Request

### Using curl

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -F "file=@/path/to/image.jpg" \
  -F "priority=standard" \
  -F "vertical=insurance"
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/verify"
files = {"file": open("image.jpg", "rb")}
data = {"priority": "standard", "vertical": "insurance"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Trust Score: {result['trust_score']}")
print(f"Risk: {result['risk_category']}")
```

### Using JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('image.jpg'));
form.append('priority', 'standard');

axios.post('http://localhost:8000/api/v1/verify', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log('Trust Score:', response.data.trust_score);
  console.log('Risk:', response.data.risk_category);
})
.catch(error => console.error(error));
```

## Understanding Results

### Trust Score
- **90-100**: Verified with cryptographic proof (C2PA)
- **70-89**: High confidence authentic
- **50-69**: Likely authentic
- **30-49**: Uncertain
- **10-29**: Likely AI-generated
- **0-9**: High confidence AI-generated or fraudulent

### Risk Categories
- `verified`: C2PA signature verified
- `authentic_high_confidence`: Very likely authentic
- `likely_authentic`: Probably authentic
- `uncertain`: Cannot determine
- `likely_synthetic`: Probably AI-generated
- `synthetic_high_confidence`: Very likely AI-generated
- `fraudulent`: Invalid signature or clear fraud

## Common Issues

### Docker not starting
```bash
# Check Docker is running
docker ps

# Check logs
docker-compose logs

# Restart services
docker-compose restart
```

### Database connection errors
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead of 8000
```

## Next Steps

1. Read the full [README.md](README.md)
2. Explore the API documentation at http://localhost:8000/docs
3. Check the [PRD](PRD.md) for complete feature details
4. Set up monitoring and logging for production

## Support

- API Docs: http://localhost:8000/docs
- Issues: GitHub Issues
- Community: Discord/Slack

---

**Ready to verify content authenticity!** 🚀
