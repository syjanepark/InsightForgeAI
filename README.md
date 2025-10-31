# InsightForge

**AI-powered business intelligence platform that analyzes uploaded datasets with real-time web intelligence to generate actionable insights for strategic decision-making.**

## 🎯 What InsightForge Does

- **📊 Dynamic Dataset Analysis**: Upload any CSV and get intelligent analysis 
- **🌐 Real-Time Context**: Enriches your data with live web intelligence using You.com Search and News APIs  
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
#   pip install -r ../requirements.txt
```

### 3. Environment Configuration
```bash
# In backend directory, create .env file:
cd backend
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

## 🏗️ Architecture

### Core Principles
- **No Hardcoding**: Column names, entity types, and data structures are detected dynamically
- **Pattern Recognition**: Automatically identifies time columns, numeric metrics, categorical entities
- **Heuristic-Based**: Uses intelligent rules instead of fixed assumptions
- **API-Driven**: Integrates with You.com Search/News for external context

### Project Structure
```
InsightForgeAI/
├── backend/
│   ├── agents/           # Analysis agents
│   │   ├── smart_chat_agent.py    # Main chat and analysis (fully dynamic)
│   │   ├── query_agent.py          # Query understanding
│   │   └── insight_agent.py        # Insight generation (local, no external AI)
│   ├── core/            # Core business logic
│   │   ├── data_store.py          # In-memory data management
│   │   ├── etl_pipeline.py        # Dynamic data cleaning (no hardcoded schemas)
│   │   └── schemas.py             # API data models
│   ├── routers/         # API endpoints
│   │   ├── analyze_endpoint.py    # CSV upload and analysis
│   │   ├── chat.py                # Chat interface API
│   │   └── chart_preview.py       # Dynamic chart generation
│   ├── services/        # External API integrations
│   │   ├── you_search.py         # You.com Search API
│   │   ├── you_news.py           # You.com News API
│   │   └── you_contents.py       # You.com Contents API
│   └── app.py          # FastAPI application entry point
├── frontend/
│   ├── src/
│   │   ├── app/         # Next.js app router
│   │   ├── components/  # React components
│   │   │   ├── dashboard/  # Chart builder, summary display
│   │   │   └── charts/    # Dynamic chart renderer
│   │   └── lib/         # Utilities and API client
│   └── ...
└── README.md
```

## 🔧 How It Works

### 1. Dynamic Analysis Pipeline

**ETL (Extract, Transform, Load):**
- Automatically detects column types (numeric, categorical, datetime)
- Handles missing values and data cleaning
- Normalizes date formats (supports multiple formats dynamically)
- No predefined schemas—works with any CSV structure

**Pattern Detection:**
- **Time Columns**: Detects date/datetime columns via pattern matching
- **Entities**: Extracts company/product names from questions dynamically
- **Metrics**: Identifies KPI-like columns (revenue, sales, volume, etc.) via keyword matching
- **Correlations**: Finds relationships between numeric columns automatically

**Smart Analysis:**
- Adapts analysis based on detected data patterns
- Generates queries dynamically based on question content
- Builds charts based on available columns (not hardcoded)

### 2. Question Understanding

The system handles natural language questions dynamically:

- **Entity Extraction**: "why did **Netflix** revenue drop" → extracts "Netflix"
- **Year Detection**: "2024 vs 2023" → extracts both years
- **Metric Detection**: "revenue drop", "sales peak" → identifies metrics
- **Trend Questions**: "over time", "trends" → triggers time-series analysis

**Example Flow:**
```
User: "why did Netflix global revenue drop in 2024 vs 2023?"
↓
Extracts: entity="Netflix", years=[2024,2023], concept="revenue"
↓
Generates queries: ["why did netflix global revenue drop in 2024 vs 2023", 
                    "netflix 2024 revenue", 
                    "netflix 2024 vs 2023 revenue"]
↓
Searches: You.com Search API (with fallback to News API)
↓
Synthesizes: Local analysis + external context
```

### 3. Web Intelligence Integration

**Search Strategy:**
1. Tries exact user question first (most reliable)
2. Falls back to heuristic-generated queries if needed
3. Uses You.com News API if Search API fails (automatic fallback)

**Dynamic Query Generation:**
- Extracts entities/years/concepts from question text
- Works for any company, product, or metric mentioned

### 4. Chart Generation

**Dynamic Chart Builder:**
- Users select chart type, X-axis, Y-axis from detected columns
- Supports: Line, Bar, Pie, Scatter (when appropriate)
- Auto-suggests sensible defaults based on data types

**Chart Summary:**
- Generates Insight/Reasoning/Implication from actual chart data
- Calculates specific metrics: start→end Δ, CAGR, peaks, shares
- Fully local analysis—no external AI dependencies

## 🔑 API Configuration

### You.com API Setup
1. Get API key from [You.com Developer Portal](https://api.you.com)
2. Enable required scopes:
   - **Search** (for web results)
   - **News** (as fallback)
3. Add to `backend/.env`:
   ```
   YDC_API_KEY=your_api_key_here
   ```
4. Restart backend server

### API Endpoints
- `POST /analyze/` - Upload CSV and get dynamic analysis
- `POST /chat/` - Send questions about uploaded data
- `POST /chart/columns` - Get available columns for chart building
- `POST /chart/preview` - Generate chart from user selections
- `POST /chart/summarize` - Get Insight/Reasoning/Implication for charts
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 🧪 Testing

### Manual Testing
```bash
# Upload a CSV file
curl -X POST -F "file=@your-data.csv" http://localhost:8000/analyze/

# Test chat functionality  
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "why did sales drop in 2024", "run_id": "your-run-id"}' \
  http://localhost:8000/chat/

# Test chart generation
curl -X POST -H "Content-Type: application/json" \
  -d '{"run_id": "your-run-id", "chart_type": "line", "x": "Date", "y": "Revenue", "agg": "sum"}' \
  http://localhost:8000/chart/preview
```

### Dynamic Testing Checklist
- ✅ Upload CSVs with different structures (various column names)
- ✅ Test questions with different entities (not just Netflix)
- ✅ Verify entity extraction works for any company/product
- ✅ Confirm year comparison queries work for any years
- ✅ Check chart builder works with any numeric/categorical columns

## 📝 Key Features

### Intelligent Fallbacks
- Search API → News API (automatic)
- Exact query → Generated queries (automatic)
- External context → Local-only analysis (graceful degradation)

### Local-First Architecture
- All synthesis is local (no external AI dependencies)
- Chart summaries generated from actual data calculations
- Faster, more reliable, no API costs for core features

## 🚨 Common Issues

### Backend Not Starting
- Check Python version: `python --version` (needs 3.9+)
- Verify virtual environment: `which python` should point to venv
- Check dependencies: `pip install -r requirements.txt`
- Verify `.env` file exists in `backend/` directory

### API Key Issues
- Ensure `YDC_API_KEY` is set in `backend/.env` (~93 characters, not 18)
- Enable "Search" scope in You.com dashboard
- Restart backend after changing `.env` (see API Configuration section above)

### Frontend Issues
- Clear browser cache and restart dev server
- Check console for JavaScript errors
- Verify backend is running on port 8000

### No Search Results
- Check You.com API key has Search scope enabled
- Verify key length in logs (should be ~93, not 18)
- System automatically falls back to News API if Search fails

## 🚀 Deployment

### Environment Variables
```bash
# Production environment
YDC_API_KEY=your_production_api_key
NODE_ENV=production
```

### Backend Deployment
```bash
cd backend
pip install -r requirements.txt
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend Deployment
```bash
npm run build
npm start
```

---

**Built for teams who need intelligent, actionable insights from their data — works with any dataset structure.** 🚀
