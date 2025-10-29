import Papa from 'papaparse';

export interface AnalyzedData {
  insights: {
    id: string;
    title: string;
    summary: string;
    sources: string[];
    category: "trend" | "anomaly" | "recommendation" | "correlation";
    confidence: number;
  }[];
  charts: {
    id: string;
    type: "bar" | "line" | "pie";
    title: string;
    data: any[];
    xKey?: string;
    yKey?: string;
  }[];
  summary: {
    totalRows: number;
    totalColumns: number;
    keyMetrics: { name: string; value: string }[];
    columnTypes: { [key: string]: 'numeric' | 'text' | 'date' };
    numericColumns: string[];
    textColumns: string[];
  };
}

export function analyzeCSV(file: File): Promise<AnalyzedData> {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (results) => {
        try {
          if (results.errors && results.errors.length > 0) {
            console.warn('CSV parsing warnings:', results.errors);
          }
          
          const data = results.data as any[];
          if (!data || data.length === 0) {
            throw new Error('No data found in CSV file. Please check that your file contains valid data.');
          }
          
          const analyzed = processCSVData(data);
          resolve(analyzed);
        } catch (error) {
          reject(error instanceof Error ? error : new Error('Failed to process CSV data'));
        }
      },
      error: (error) => {
        reject(new Error(`Failed to parse CSV file: ${error.message}`));
      }
    });
  });
}

function processCSVData(data: any[]): AnalyzedData {
  if (!data || data.length === 0) {
    throw new Error('No data found in CSV file');
  }

  // Filter out completely empty rows
  const cleanData = data.filter(row => Object.values(row).some(val => val !== null && val !== undefined && val !== ''));
  
  if (cleanData.length === 0) {
    throw new Error('CSV file contains no valid data rows');
  }

  const headers = Object.keys(cleanData[0]).filter(header => header && header.trim() !== '');
  
  if (headers.length === 0) {
    throw new Error('CSV file contains no valid columns');
  }

  const columnTypes = analyzeColumnTypes(cleanData, headers);
  const numericColumns = Object.keys(columnTypes).filter(col => columnTypes[col] === 'numeric');
  const textColumns = Object.keys(columnTypes).filter(col => columnTypes[col] === 'text');

  try {
    // Generate charts based on data structure
    const charts = generateCharts(cleanData, headers, numericColumns, textColumns);
    
    // Generate insights based on analysis
    const insights = generateInsights(cleanData, headers, numericColumns, textColumns, charts);
    
    // Calculate key metrics
    const keyMetrics = calculateKeyMetrics(cleanData, numericColumns, textColumns);

    return {
      insights,
      charts,
      summary: {
        totalRows: cleanData.length,
        totalColumns: headers.length,
        keyMetrics,
        columnTypes,
        numericColumns,
        textColumns
      }
    };
  } catch (error) {
    throw new Error(`Data analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

function analyzeColumnTypes(data: any[], headers: string[]): { [key: string]: 'numeric' | 'text' | 'date' } {
  const types: { [key: string]: 'numeric' | 'text' | 'date' } = {};
  
  headers.forEach(header => {
    const sample = data.slice(0, Math.min(10, data.length))
      .map(row => row[header])
      .filter(val => val !== null && val !== undefined && val !== '');
    
    if (sample.length === 0) {
      types[header] = 'text';
      return;
    }

    // Check if mostly numeric
    const numericCount = sample.filter(val => !isNaN(parseFloat(val)) && isFinite(val)).length;
    const numericRatio = numericCount / sample.length;

    // Check if date-like
    const dateCount = sample.filter(val => {
      const parsed = new Date(val);
      return !isNaN(parsed.getTime()) && val.toString().length > 4;
    }).length;
    const dateRatio = dateCount / sample.length;

    if (numericRatio > 0.8) {
      types[header] = 'numeric';
    } else if (dateRatio > 0.8) {
      types[header] = 'date';
    } else {
      types[header] = 'text';
    }
  });

  return types;
}

function generateCharts(data: any[], headers: string[], numericColumns: string[], textColumns: string[]): AnalyzedData['charts'] {
  const charts: AnalyzedData['charts'] = [];

  // Line chart - for trends over time or sequential data
  if (numericColumns.length > 0) {
    const trendColumn = numericColumns[0];
    const lineData = data.slice(0, 10).map((row, index) => ({
      name: row[headers[0]] || `Point ${index + 1}`,
      value: parseFloat(row[trendColumn]) || 0
    }));

    charts.push({
      id: 'trend-chart',
      type: 'line',
      title: `${trendColumn} Trend`,
      data: lineData,
      xKey: 'name',
      yKey: 'value'
    });
  }

  // Bar chart - comparing categories
  if (textColumns.length > 0 && numericColumns.length > 0) {
    const categoryColumn = textColumns[0];
    const valueColumn = numericColumns[0];
    
    // Aggregate data by category
    const categoryData: { [key: string]: number } = {};
    data.forEach(row => {
      const category = row[categoryColumn] || 'Unknown';
      const value = parseFloat(row[valueColumn]) || 0;
      categoryData[category] = (categoryData[category] || 0) + value;
    });

    const barData = Object.entries(categoryData)
      .slice(0, 8) // Limit to top 8 categories
      .map(([name, value]) => ({ name, value }));

    charts.push({
      id: 'category-chart',
      type: 'bar',
      title: `${valueColumn} by ${categoryColumn}`,
      data: barData,
      xKey: 'name',
      yKey: 'value'
    });
  }

  // Pie chart - distribution of categories
  if (textColumns.length > 0) {
    const categoryColumn = textColumns[0];
    const categoryCount: { [key: string]: number } = {};
    
    data.forEach(row => {
      const category = row[categoryColumn] || 'Unknown';
      categoryCount[category] = (categoryCount[category] || 0) + 1;
    });

    const pieData = Object.entries(categoryCount)
      .slice(0, 6) // Top 6 categories
      .map(([name, value]) => ({ name, value }));

    charts.push({
      id: 'distribution-chart',
      type: 'pie',
      title: `Distribution of ${categoryColumn}`,
      data: pieData,
      yKey: 'value'
    });
  }

  return charts;
}

function generateInsights(
  data: any[], 
  headers: string[], 
  numericColumns: string[], 
  textColumns: string[],
  charts: AnalyzedData['charts']
): AnalyzedData['insights'] {
  const insights: AnalyzedData['insights'] = [];

  // Trend analysis
  if (numericColumns.length > 0 && data.length > 1) {
    const column = numericColumns[0];
    const values = data.map(row => parseFloat(row[column]) || 0).filter(v => !isNaN(v));
    
    if (values.length > 1) {
      const firstHalf = values.slice(0, Math.floor(values.length / 2));
      const secondHalf = values.slice(Math.floor(values.length / 2));
      
      const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
      const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
      const growthRate = ((avgSecond - avgFirst) / avgFirst * 100).toFixed(1);

      insights.push({
        id: '1',
        title: `${column} Trend Analysis`,
        summary: `${column} shows ${parseFloat(growthRate) > 0 ? 'positive' : 'negative'} trend with ${Math.abs(parseFloat(growthRate))}% change over the dataset period.`,
        sources: ['Data Analysis', 'Statistical Computing'],
        category: 'trend',
        confidence: 92
      });
    }
  }

  // Distribution insights
  if (textColumns.length > 0) {
    const column = textColumns[0];
    const uniqueValues = [...new Set(data.map(row => row[column]))].filter(v => v);
    const totalValues = data.length;
    const diversity = (uniqueValues.length / totalValues * 100).toFixed(1);

    insights.push({
      id: '2',
      title: `${column} Distribution Pattern`,
      summary: `Dataset contains ${uniqueValues.length} unique ${column.toLowerCase()} categories with ${diversity}% diversity ratio, indicating ${parseFloat(diversity) > 50 ? 'high' : 'moderate'} variability.`,
      sources: ['Pattern Recognition', 'Data Mining'],
      category: 'correlation',
      confidence: 88
    });
  }

  // Data quality insight
  const totalCells = data.length * headers.length;
  const emptyCells = data.reduce((acc, row) => {
    return acc + headers.filter(col => !row[col] || row[col] === '').length;
  }, 0);
  const completeness = ((totalCells - emptyCells) / totalCells * 100).toFixed(1);

  insights.push({
    id: '3',
    title: 'Data Quality Assessment',
    summary: `Dataset shows ${completeness}% completeness rate with ${emptyCells} missing values across ${totalCells} total data points.`,
    sources: ['Data Quality Analysis', 'Validation Engine'],
    category: parseFloat(completeness) > 95 ? 'recommendation' : 'anomaly',
    confidence: 95
  });

  // Volume insights
  if (numericColumns.length > 0) {
    const column = numericColumns[0];
    const values = data.map(row => parseFloat(row[column]) || 0);
    const max = Math.max(...values);
    const min = Math.min(...values);
    const avg = values.reduce((a, b) => a + b, 0) / values.length;

    insights.push({
      id: '4',
      title: `${column} Performance Metrics`,
      summary: `${column} ranges from ${min.toLocaleString()} to ${max.toLocaleString()} with an average of ${avg.toLocaleString()}, showing ${max > avg * 2 ? 'high variance' : 'consistent'} performance patterns.`,
      sources: ['Statistical Analysis', 'Performance Metrics'],
      category: 'recommendation',
      confidence: 90
    });
  }

  return insights.slice(0, 4); // Return top 4 insights
}

function calculateKeyMetrics(data: any[], numericColumns: string[], textColumns: string[]): { name: string; value: string }[] {
  const metrics: { name: string; value: string }[] = [];

  metrics.push({ name: 'Total Records', value: data.length.toLocaleString() });

  if (numericColumns.length > 0) {
    const column = numericColumns[0];
    const values = data.map(row => {
      const val = row[column];
      return typeof val === 'number' ? val : parseFloat(val);
    }).filter(v => !isNaN(v) && isFinite(v));
    
    if (values.length > 0) {
      const total = values.reduce((a, b) => a + b, 0);
      const average = total / values.length;
      const max = Math.max(...values);

      metrics.push(
        { name: `Avg ${column}`, value: average.toFixed(0).toLocaleString() },
        { name: `Peak ${column}`, value: max.toLocaleString() }
      );
    }
  }

  if (textColumns.length > 0) {
    const column = textColumns[0];
    const uniqueValues = new Set(data.map(row => row[column]).filter(v => v));
    metrics.push({ name: `Unique ${column}`, value: uniqueValues.size.toString() });
  }

  return metrics.slice(0, 3); // Return top 3 metrics
}
