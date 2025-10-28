"use client";

import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = ['#8B7CF6', '#A78BFA', '#60A5FA', '#34D399', '#F59E0B', '#EF4444'];

interface ChartData {
  [key: string]: any;
}

interface ChartProps {
  title: string;
  data: ChartData[];
  type: 'line' | 'bar' | 'pie';
  xKey?: string;
  yKey?: string;
  delay?: number;
}

export function Chart({ title, data, type, xKey = 'name', yKey = 'value', delay = 0 }: ChartProps) {
  if (!data || data.length === 0) {
    return (
      <div 
        className="glass-card p-6 hover-lift slide-up-fade"
        style={{ animationDelay: `${delay}ms` }}
      >
        <h3 className="text-lg font-semibold text-deep-indigo mb-4">
          {title}
        </h3>
        <div className="h-64 bg-gradient-to-br from-primary/5 to-secondary/5 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center animate-pulse">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p className="text-sm text-deep-indigo/60">Processing your data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="glass-card p-6 hover-lift slide-up-fade"
      style={{ animationDelay: `${delay}ms` }}
    >
      <h3 className="text-lg font-semibold text-deep-indigo mb-4">
        {title}
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'line' && (
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 124, 246, 0.1)" />
              <XAxis 
                dataKey={xKey} 
                stroke="#312E81" 
                fontSize={12}
                tick={{ fill: '#312E81' }}
              />
              <YAxis 
                stroke="#312E81" 
                fontSize={12}
                tick={{ fill: '#312E81' }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  border: '1px solid rgba(139, 124, 246, 0.2)',
                  borderRadius: '12px',
                  boxShadow: '0 8px 25px rgba(139, 124, 246, 0.15)',
                  backdropFilter: 'blur(10px)'
                }}
              />
              <Line 
                type="monotone" 
                dataKey={yKey} 
                stroke="#8B7CF6" 
                strokeWidth={3}
                dot={{ fill: '#8B7CF6', strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, fill: '#A78BFA' }}
              />
            </LineChart>
          )}

          {type === 'bar' && (
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 124, 246, 0.1)" />
              <XAxis 
                dataKey={xKey} 
                stroke="#312E81" 
                fontSize={12}
                tick={{ fill: '#312E81' }}
              />
              <YAxis 
                stroke="#312E81" 
                fontSize={12}
                tick={{ fill: '#312E81' }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  border: '1px solid rgba(139, 124, 246, 0.2)',
                  borderRadius: '12px',
                  boxShadow: '0 8px 25px rgba(139, 124, 246, 0.15)',
                  backdropFilter: 'blur(10px)'
                }}
              />
              <Bar dataKey={yKey} radius={[4, 4, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          )}

          {type === 'pie' && (
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }: any) => `${name} ${(Number(percent || 0) * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey={yKey}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  border: '1px solid rgba(139, 124, 246, 0.2)',
                  borderRadius: '12px',
                  boxShadow: '0 8px 25px rgba(139, 124, 246, 0.15)',
                  backdropFilter: 'blur(10px)'
                }}
              />
            </PieChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function LineChartComponent({ title, data, xKey, yKey, delay }: Omit<ChartProps, 'type'>) {
  return <Chart title={title} data={data} type="line" xKey={xKey} yKey={yKey} delay={delay} />;
}

export function BarChartComponent({ title, data, xKey, yKey, delay }: Omit<ChartProps, 'type'>) {
  return <Chart title={title} data={data} type="bar" xKey={xKey} yKey={yKey} delay={delay} />;
}

export function PieChartComponent({ title, data, yKey, delay }: Omit<ChartProps, 'type' | 'xKey'>) {
  return <Chart title={title} data={data} type="pie" yKey={yKey} delay={delay} />;
}

