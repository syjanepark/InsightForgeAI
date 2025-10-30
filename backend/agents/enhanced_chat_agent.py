# agents/enhanced_chat_agent.py
from services.you_smart import call_smart
from services.you_search import search_web
import pandas as pd
import asyncio
import re
import json

class EnhancedChatAgent:
    def __init__(self):
        self.smart_agent = None  # Will be initialized with LLM service
    
    async def process_query(self, question: str, context: str, df: pd.DataFrame = None):
        """Enhanced query processing with LLM-powered entity detection"""
        
        # Use LLM to extract entities and determine if search is needed
        entity_analysis = await self._extract_entities_with_llm(question, df)
        needs_search = entity_analysis.get('needs_external_search', False)
        entities = entity_analysis.get('entities', [])
        years = entity_analysis.get('years', [])
        concepts = entity_analysis.get('concepts', [])
        
        print(f"🤖 LLM Analysis: entities={entities}, years={years}, concepts={concepts}, needs_search={needs_search}")
        
        search_context = ""
        if needs_search:
            print(f"🔍 LLM detected need for external context, searching...")
            search_results = await self._get_contextual_search(question, entities, years, concepts)
            if search_results:
                search_context = f"\n\nEXTERNAL CONTEXT:\n{search_results}"
                print(f"✅ Found external context: {len(search_context)} chars")
        
        # Enhanced prompt with LLM-extracted context
        prompt = f"""
You are a senior business analyst with access to both internal data and real-time market intelligence.

DATA CONTEXT:
{context}
{search_context}

QUESTION: {question}

EXTRACTED ENTITIES: {entities}
EXTRACTED YEARS: {years}
EXTRACTED CONCEPTS: {concepts}

REQUIREMENTS:
- Analyze the actual data provided in the context
- Be specific and cite exact metrics/columns from the dataset
- Provide actionable insights based on the real data
- If external context was found, reference it to explain "why" behind data patterns
- Format response in structured Insight/Reasoning/Implication format when external context is available
"""
        
        try:
            out = call_smart(prompt, use_tools=True)
            if out and "Request failed" not in out:
                return {
                    "answer": out,
                    "visualizations": None,
                    "suggested_actions": None,
                    "citations": self._extract_citations(out)
                }
        except Exception as e:
            print(f"🔄 You.com API failed, using local analysis: {e}")
        
        # Fallback to local analysis
        local_answer = self._create_local_analysis(question, context, df)
        return {
            "answer": local_answer,
            "visualizations": None,
            "suggested_actions": None,
            "citations": None
        }
    
    async def _extract_entities_with_llm(self, question: str, df: pd.DataFrame = None) -> dict:
        """Use LLM to intelligently extract entities and determine search needs"""
        
        # Create context about the dataset
        dataset_context = ""
        if df is not None:
            dataset_context = f"""
Dataset has {len(df)} rows and {len(df.columns)} columns.
Columns: {', '.join(df.columns.tolist())}
Sample data types: {dict(df.dtypes)}
"""
        
        # LLM prompt for entity extraction
        entity_prompt = f"""
Analyze this question and extract relevant information for data analysis:

QUESTION: "{question}"

DATASET CONTEXT:
{dataset_context}

Please extract and return a JSON response with:
1. "entities": List of company names, products, or key entities mentioned (case-insensitive)
2. "years": List of years mentioned (4-digit format)
3. "concepts": List of business/financial concepts (sales, revenue, growth, etc.)
4. "needs_external_search": Boolean indicating if this question needs external web context
5. "reasoning": Brief explanation of why external search is/isn't needed

Guidelines:
- Extract entities even if not capitalized (e.g., "apple" from "why did apple stock drop")
- Look for patterns like "why did [entity]...", "[entity] stock", "[entity] in [year]"
- External search needed for: "why" questions, "what caused" questions, market explanations
- External search NOT needed for: data summaries, correlations, basic statistics

Return only valid JSON, no other text.
"""
        
        try:
            # Use You.com Smart API for entity extraction
            response = call_smart(entity_prompt, use_tools=False)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                entity_data = json.loads(json_match.group())
                return entity_data
            else:
                # Fallback to regex-based extraction
                return self._fallback_entity_extraction(question)
                
        except Exception as e:
            print(f"⚠️ LLM entity extraction failed: {e}")
            return self._fallback_entity_extraction(question)
    
    def _fallback_entity_extraction(self, question: str) -> dict:
        """Fallback to regex-based extraction if LLM fails"""
        question_lower = question.lower()
        
        # Extract years
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
        
        # Extract entities using patterns
        entities = []
        all_words = re.findall(r'\b[a-zA-Z]+\b', question_lower)
        stop_words = {
            'why', 'what', 'when', 'where', 'how', 'the', 'this', 'that', 'these', 'those',
            'did', 'does', 'do', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'stock', 'market', 'company', 'firm', 'revenue', 'sales', 'profit', 'earnings',
            'growth', 'decline', 'crash', 'boom', 'performance', 'prices', 'value'
        }
        
        for word in all_words:
            if word not in stop_words and len(word) > 2:
                if re.search(rf'\b(why did|what caused|what happened to|how did)\s+{re.escape(word)}\b', question_lower):
                    entities.append(word)
                elif re.search(rf'\b{re.escape(word)}\s+(stock|revenue|sales|growth|decline|crash|boom|performance)\b', question_lower):
                    entities.append(word)
                elif re.search(rf'\b{re.escape(word)}\s+in\s+\d{{4}}\b', question_lower):
                    entities.append(word)
        
        entities = list(dict.fromkeys(entities))
        
        # Extract concepts
        concepts = []
        concept_patterns = [
            r'\b(sales|revenue|growth|decline|drop|increase|decrease)\b',
            r'\b(performance|volatility|trading|volume|price|value)\b',
            r'\b(profit|loss|earnings|income|expense)\b',
            r'\b(market|economy|financial|economic)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, question_lower)
            concepts.extend(matches)
        
        # Determine if external search is needed
        needs_search = any(keyword in question_lower for keyword in [
            'why', 'because', 'reason', 'cause', 'due to', 'behind',
            'what caused', 'how come', 'explain', 'what happened',
            'what led to', 'what drove', 'what influenced', 'what affected'
        ])
        
        return {
            'entities': entities,
            'years': years,
            'concepts': concepts,
            'needs_external_search': needs_search,
            'reasoning': 'Fallback regex-based extraction'
        }
    
    async def _get_contextual_search(self, question: str, entities: list, years: list, concepts: list) -> str:
        """Enhanced search with LLM-generated queries"""
        
        # Generate smart search queries using LLM
        search_queries = await self._generate_llm_search_queries(question, entities, years, concepts)
        
        try:
            # Perform searches
            loop = asyncio.get_event_loop()
            search_tasks = [
                loop.run_in_executor(None, search_web, query, 2) 
                for query in search_queries
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and format results
            combined_results = []
            for result_group in results:
                if isinstance(result_group, list):
                    combined_results.extend(result_group)
            
            if not combined_results:
                return ""
            
            # Format search results
            formatted_context = []
            for i, result in enumerate(combined_results[:4], 1):
                title = result.get('title', 'Unknown')
                snippet = result.get('snippet', '')
                url = result.get('url', '')
                
                if snippet:
                    formatted_context.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")
            
            return '\n\n'.join(formatted_context) if formatted_context else ""
            
        except Exception as e:
            print(f"⚠️ Search failed: {e}")
            return ""
    
    async def _generate_llm_search_queries(self, question: str, entities: list, years: list, concepts: list) -> list:
        """Use LLM to generate smart search queries"""
        
        query_prompt = f"""
Generate 2-3 targeted web search queries for this question:

QUESTION: "{question}"
ENTITIES: {entities}
YEARS: {years}
CONCEPTS: {concepts}

Create specific, focused search queries that will find relevant external context.
Focus on:
- Company-specific analysis if entities are mentioned
- Year-specific events if years are mentioned
- Market/financial context for business questions

Return as a JSON array of strings:
["query1", "query2", "query3"]
"""
        
        try:
            response = call_smart(query_prompt, use_tools=False)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                return queries[:3]  # Limit to 3 queries
        except Exception as e:
            print(f"⚠️ LLM query generation failed: {e}")
        
        # Fallback to simple queries
        fallback_queries = []
        if entities and years:
            fallback_queries.append(f"{entities[0]} {years[0]} analysis")
        if entities:
            fallback_queries.append(f"{entities[0]} market analysis")
        if years:
            fallback_queries.append(f"{years[0]} market trends")
        
        return fallback_queries if fallback_queries else [question + " analysis"]
    
    def _create_local_analysis(self, question: str, context: str, df: pd.DataFrame = None) -> str:
        """Create local analysis fallback"""
        if df is None:
            return "I need access to the raw data to answer specific questions. Please re-upload your CSV file."
        
        return f"Based on your dataset with {len(df)} rows and {len(df.columns)} columns, here's what I can tell you about: {question}"
    
    def _extract_citations(self, text: str) -> list:
        """Extract citations from text"""
        if not text or "Source:" not in text:
            return None
        
        citations = []
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith('Source:'):
                url = line.replace('Source:', '').strip()
                if url:
                    citations.append(url)
        
        return citations if citations else None
