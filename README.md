# InsightForge

**AI-powered business intelligence platform that analyzes uploaded datasets with real-time web intelligence to generate actionable insights for strategic decision-making.**

## 🎯 What InsightForge Does

- **📊 Dynamic Dataset Analysis**: Upload any CSV and get intelligent analysis without hardcoded assumptions
- **🌐 Real-Time Context**: Enriches your data with live web intelligence using You.com APIs  
- **🧠 Smart Reasoning**: Advanced pattern recognition and contextual analysis for business insights
- **📈 Actionable Reports**: Clear visualizations and strategic recommendations
- **⚡ Instant Results**: Fast analysis pipeline designed for business teams who need answers now

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.9+ 
- **You.com API Key** (for web intelligence features)

### 1. Clone & Setup
```bash
git clone <repository-url>
cd InsightForgeAI
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# In backend directory, create .env file:
echo "YDC_API_KEY=your_you_com_api_key_here" > .env
```

### 4. Frontend Setup
```bash
cd ../frontend
npm install
```

### 5. Run Development Servers

**Backend (Terminal 1):**
```bash
cd backend
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend (Terminal 2):**
```bash
cd frontend
npm run dev
```

🎉 **Access the app**: http://localhost:3000

## 🏗️ Project Structure

```
InsightForgeAI/
├── backend/
│   ├── agents/           # AI agents for different analysis types
│   │   ├── smart_chat_agent.py    # Main chat and analysis logic
│   │   └── ...
│   ├── core/            # Core business logic
│   │   ├── data_store.py          # In-memory data management
│   │   ├── etl_pipeline.py        # Data cleaning and processing
│   │   └── schemas.py             # API data models
│   ├── routers/         # API endpoints
│   │   ├── analyze_endpoint.py    # CSV upload and analysis
│   │   └── chat.py               # Chat interface API
│   ├── services/        # External API integrations
│   │   ├── you_search.py         # You.com search API
│   │   └── you_smart.py          # You.com advanced agent API
│   └── app.py          # FastAPI application entry point
├── frontend/
│   ├── src/
│   │   ├── app/         # Next.js app router
│   │   ├── components/  # React components
│   │   └── lib/         # Utilities and API client
│   └── ...
└── README.md
```

## 🔧 Development Workflow

### Key Principles
- **No Hardcoding**: All analysis is dynamic and works with any CSV structure
- **Generic Patterns**: Code detects data patterns rather than assumes specific columns
- **Smart Defaults**: Intelligent column detection (time, entity, value columns)

### Making Changes

1. **Backend Changes**: 
   - Modify agents for analysis logic
   - Update routers for API endpoints
   - Test with `curl` commands or FastAPI docs at http://localhost:8000/docs

2. **Frontend Changes**:
   - Components in `src/components/`
   - Main app in `src/app/(home)/page.tsx`
   - Hot reload enabled in development

### Testing Analysis Logic

```bash
# Upload a CSV file
curl -X POST -F "file=@your-data.csv" http://localhost:8000/analyze/

# Test chat functionality  
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "what year did sales peak"}' \
  http://localhost:8000/chat/
```

## 🧠 How It Works

### 1. Dynamic Analysis Pipeline
- **ETL**: Automatically cleans and normalizes any CSV structure
- **Pattern Detection**: Identifies entity, time, and value columns dynamically
- **Smart Analysis**: Adapts analysis based on detected data patterns

### 2. Question Understanding
The system handles natural language questions like:
- "top sales by each year" → Time-series analysis
- "what year did [entity] peak" → Peak detection analysis  
- "trends over time" → Trend analysis

### 3. Web Intelligence Integration
- Contextual questions trigger web searches for background information
- You.com APIs provide real-time market context
- Results combine internal data analysis with external insights

## 🔑 API Configuration

### You.com API Setup
1. Get API key from [You.com Developer Portal](https://api.you.com)
2. Add to backend `.env` file:
   ```
   YDC_API_KEY=your_api_key_here
   ```
3. Restart backend server

### API Endpoints
- `POST /analyze/` - Upload CSV and get analysis
- `POST /chat/` - Send questions about uploaded data
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 🤝 Contributing

### Branch Strategy
- `main` - Production-ready code
- `feature/*` - New features
- `fix/*` - Bug fixes

### Commit Guidelines
- Use descriptive commit messages
- Keep commits focused and atomic
- Reference issues where applicable

### Code Style
- **Backend**: Follow PEP 8 for Python
- **Frontend**: ESLint configuration provided
- **No Hardcoding**: Always use dynamic analysis patterns

## 🔬 Testing

### Backend Testing
```bash
cd backend
python -m pytest test_*.py
```

### Frontend Testing
```bash
cd frontend  
npm test
```

### Manual Testing
1. Upload various CSV formats
2. Test different question patterns
3. Verify dynamic analysis works across datasets

## 📝 Common Issues

### Backend Not Starting
- Check Python version: `python --version` (needs 3.9+)
- Verify virtual environment: `which python` should point to venv
- Check dependencies: `pip install -r requirements.txt`

### API Errors
- Verify YDC_API_KEY is set correctly
- Check backend logs for specific error messages
- Test basic endpoints: `curl http://localhost:8000/health`

### Frontend Issues
- Clear browser cache and restart dev server
- Check console for JavaScript errors
- Verify backend is running on port 8000

## 🚀 Deployment

### Environment Variables
```bash
# Production environment
YDC_API_KEY=your_production_api_key
NODE_ENV=production
```

### Backend Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production server
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend Deployment
```bash
npm run build
npm start
```


**Built for teams who need intelligent, actionable insights from their data — not just charts and numbers.** 🚀