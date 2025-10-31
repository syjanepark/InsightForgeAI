# services/claude_refinement.py
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

class ClaudeRefinementService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        # Allow overriding model via env; default to user-preferred stable name
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
    
    def refine_analysis(self, base_answer: str, question: str, context_type: str = "business") -> str:
        """
        Refine You.com's analysis with Claude for better presentation and clarity
        
        Args:
            base_answer: Raw answer from You.com APIs
            question: Original user question
            context_type: Type of context (business, technical, executive, etc.)
        """
        
        if not self.client:
            return base_answer  # Fallback if Claude not available
        
        # Determine refinement approach based on answer length and complexity
        if len(base_answer) < 200:
            return base_answer  # Don't refine short, simple answers
        
        # Create context-aware refinement prompt
        refinement_prompt = self._create_refinement_prompt(base_answer, question, context_type)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user", 
                        "content": refinement_prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"⚠️ Claude refinement failed: {e}")
            return base_answer  # Return original if refinement fails
    
    def _create_refinement_prompt(self, base_answer: str, question: str, context_type: str) -> str:
        """Create context-aware refinement prompt"""
        
        if context_type == "executive":
            return f"""
You are a senior business analyst preparing an executive summary. 

The following analysis comes from You.com APIs with real-time data and web context. Do NOT change any facts, numbers, or data points.

Improve the presentation for executive-level clarity and actionability. Structure as:

**Key Insight:** [Main finding in 1-2 sentences]

**Analysis:** [Supporting evidence and reasoning]

**Strategic Implication:** [What this means for decision-making]

Original Analysis:
{base_answer}

Question: {question}
"""

        elif context_type == "technical":
            return f"""
You are a data analyst presenting technical findings.

The following analysis comes from You.com APIs with real-time data and web context. Do NOT change any facts, numbers, or data points.

Improve technical clarity while maintaining precision. Include relevant metrics and data references.

Original Analysis:
{base_answer}

Question: {question}
"""

        else:  # business context
            return f"""
You are a business analyst presenting insights to stakeholders.

The following analysis comes from You.com APIs with real-time data and web context. Do NOT change any facts, numbers, or data points.

Improve clarity, flow, and business relevance. Structure as:

**Insight:** [What the data shows]

**Reasoning:** [Why this happened - combine data evidence with external context]

**Implication:** [What this means for business decisions]

Original Analysis:
{base_answer}

Question: {question}
"""
    
    def should_refine(self, base_answer: str, question: str) -> bool:
        """
        Determine if the answer should be refined by Claude
        
        Returns True for:
        - Long answers (>200 chars)
        - Complex reasoning questions
        - Executive summary requests
        - Strategic analysis requests
        """
        
        # Always refine long answers
        if len(base_answer) > 200:
            return True
        
        # Refine complex reasoning questions
        reasoning_keywords = [
            'why', 'what caused', 'explain', 'analyze', 'strategy', 
            'implication', 'recommendation', 'forecast', 'trend'
        ]
        
        if any(keyword in question.lower() for keyword in reasoning_keywords):
            return True
        
        # Refine executive/strategic requests
        executive_keywords = [
            'executive', 'summary', 'overview', 'strategy', 'recommendation',
            'implication', 'action', 'decision', 'plan'
        ]
        
        if any(keyword in question.lower() for keyword in executive_keywords):
            return True
        
        return False
    
    def get_context_type(self, question: str, base_answer: str) -> str:
        """Determine the appropriate context type for refinement"""
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['executive', 'summary', 'overview', 'strategy']):
            return "executive"
        elif any(word in question_lower for word in ['technical', 'data', 'metrics', 'statistics', 'correlation']):
            return "technical"
        else:
            return "business"

    def rank_kpis(self, columns: list, preview_stats: dict) -> list:
        """Ask Claude to rank columns by KPI-likelihood. Returns ordered list of column names.
        preview_stats: {col: {dtype, missing_pct, sample_mean, sample_std}}"""
        if not self.client:
            return []
        try:
            prompt = (
                "You are a data analyst. Rank the provided columns by likelihood of being a business KPI "
                "(sales, units, revenue, volume, qty, amount), then other outcome metrics (engagement). "
                "Deprioritize specification columns (transmission, engine, color, doors, trim, drivetrain, fuel). "
                "Consider names and preview stats. Return JSON array of column names only.\n\n"
                f"COLUMNS: {columns}\n\nPREVIEW_STATS: {preview_stats}\n\n"
                "Return strictly JSON array, no commentary."
            )
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0.1,
                messages=[{"role":"user","content":prompt}]
            )
            text = resp.content[0].text if resp and resp.content else "[]"
            import json
            import re
            m = re.search(r"\[.*\]", text, re.DOTALL)
            return json.loads(m.group(0)) if m else []
        except Exception as e:
            print(f"⚠️ Claude KPI ranking failed: {e}")
            return []
