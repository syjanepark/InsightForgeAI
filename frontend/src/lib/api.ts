const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface AnalysisResult {
  run_id: string;
  charts: Array<{
    type: string;
    spec: {
      title: string;
      data: any;
    };
  }>;
  insights: Array<{
    title: string;
    why: string;
    recommendations: string[];
  }>;
  kpis: Array<{
    name: string;
    value: number;
    change: number;
  }>;
}

export interface ChatRequest {
  question: string;
  run_id?: string;
}

export interface ChatResponse {
  answer: string;
  visualizations?: Array<{
    type: string;
    spec: any;
  }>;
  suggested_actions?: Array<{
    label: string;
    action: string;
  }>;
  citations?: string[];
}

export async function analyzeCSV(file: File): Promise<AnalysisResult> {
  console.log('🔄 Creating FormData with file:', file.name, 'Size:', file.size);
  const formData = new FormData();
  formData.append('file', file);

  console.log('🌐 Making fetch request to:', `${API_BASE_URL}/analyze/`);
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 75000); // 75 second timeout for advanced analysis

  try {
    const response = await fetch(`${API_BASE_URL}/analyze/`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    console.log('📥 Got response:', response.status, response.statusText);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ API Error Response:', errorText);
      throw new Error(`Analysis failed: ${response.status} ${errorText}`);
    }

    const result = await response.json();
    console.log('✅ Parsed JSON response:', result);
    return result;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      console.error('⏰ Request timed out after 75 seconds');
      throw new Error('Advanced analysis timed out - this dataset might be too complex for real-time processing');
    }
    console.error('💥 API call failed:', error);
    throw error;
  }
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Chat failed: ${response.status} ${errorText}`);
  }

  return response.json();
}

// Data Explorer APIs
export async function getChartColumns(runId: string): Promise<{ numeric: string[]; categorical: string[]; datetime: string[] }> {
  const response = await fetch(`${API_BASE_URL}/chart/columns?run_id=${encodeURIComponent(runId)}`);
  if (!response.ok) throw new Error(`Columns failed: ${response.status}`);
  return response.json();
}

export async function getChartPreview(params: { run_id: string; x: string; y: string; agg?: string; chart_type?: 'bar'|'line'|'scatter'|'pie' }): Promise<{ type: string; spec: any }> {
  const response = await fetch(`${API_BASE_URL}/chart/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!response.ok) throw new Error(`Preview failed: ${response.status}`);
  return response.json();
}

export async function getDistinctValues(runId: string, column: string, limit: number = 50): Promise<{ values: string[]; counts?: number[] }> {
  const response = await fetch(`${API_BASE_URL}/chart/distinct?run_id=${encodeURIComponent(runId)}&column=${encodeURIComponent(column)}&limit=${limit}`);
  if (!response.ok) throw new Error(`Distinct failed: ${response.status}`);
  return response.json();
}

export async function summarizeCharts(params: { run_id: string; charts: Array<{ type: string; spec: any }> }): Promise<{ answer: string }> {
  const response = await fetch(`${API_BASE_URL}/chart/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!response.ok) throw new Error(`Summarize failed: ${response.status}`);
  return response.json();
}
