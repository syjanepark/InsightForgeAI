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
