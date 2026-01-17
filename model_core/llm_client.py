from openai import OpenAI
import os
import json
import re

class LLMClient:
    """
    Client for interacting with the LLM (Azure OpenAI / DeepSeek).
    Responsible for generating, refining, and analyzing alpha factors.
    """
    def __init__(self):
        # Configuration provided by user
        self.endpoint = "https://azureaistudioj7482241432.services.ai.azure.com/openai/v1/"
        self.api_key = "e2199c8638d749c8b92c63f91c532725"
        self.deployment_name = "DeepSeek-V3.2-Speciale"
        
        # Initialize Client
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key
        )

    def _query_llm(self, messages, temperature=0.7):
        try:
            completion = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def generate_initial_hypothesis(self, n=1):
        """Generates a starting set of alpha factors."""
        prompt = """
        You are an expert quantitative analyst.
        Your task is to propose valid Alpha Factor formulas for the Chinese A-share market.
        
        Available Data: open, close, high, low, volume, amount, vwap
        Available Operators: 
        - TimeSeries: ts_mean(x,d), ts_std(x,d), ts_sum(x,d), ts_rank(x,d), ts_delta(x,d), ts_corr(x,y,d)
        - CrossSectional: rank(x)
        - Basic: +, -, *, /, log(x), abs(x), sign(x)
        
        Format: ONLY output the formula string in a JSON list.
        Example: ["-1 * ts_rank(close, 10)", "rank(open / close)"]
        """
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Propose {n} novel alpha factors based on price-volume correlations."}
        ]
        
        response = self._query_llm(messages)
        return self._extract_formulas(response)

    def refine_factor(self, current_formula, reason, feedback_metrics):
        """
        Refines a factor based on feedback.
        Step 3 & 4 in Design: Suggest -> Code
        """
        prompt = f"""
        You are refining a quantitative factor.
        Current Formula: {current_formula}
        Performance Metrics: {json.dumps(feedback_metrics)}
        Optimize Goal: {reason}
        
        Task:
        1. Analyze why the factor has this performance issue.
        2. Suggest a mathematical transformation to fix it.
        3. Output the NEW formula wrapped in specific XML tags: <formula>YOUR_FORMULA_HERE</formula>.
        
        Example: <formula>ts_rank(close, 20)</formula>
        """
        
        messages = [
            {"role": "system", "content": "You are a coding assistant. Your final output must always be a single formula wrapped in <formula> tags."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._query_llm(messages, temperature=0.4)
        if not response: return None
                
        # Remove thinking traces first
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # 1. Try strict tag extraction - find ALL occurrences
        matches = re.findall(r'<formula>(.*?)</formula>', response, re.DOTALL)
        if matches:
            # Return the last match as it's likely the final answer
            # Filter out matches that are just " tags" or instruction repeats
            valid_matches = [m for m in matches if "YOUR_FORMULA_HERE" not in m and len(m.strip()) > 3]
            if valid_matches:
                return self._clean_string(valid_matches[-1])
            
        # 2. Try open tag extraction (if model forgot to close) - Be Stricter
        # Look for <formula> followed by typical alpha characters
        match = re.search(r'<formula>\s*(.*)', response, re.DOTALL)
        if match:
            # Take content until next newline or logical end
            content = match.group(1)
            # If it contains multiple lines, take the first non-empty one? 
            # Or assume formula is on the same line or block.
            # Let's split by newline and take the first valid-looking chunk.
            idx = content.find('\n')
            if idx != -1:
                content = content[:idx]
                
            # Strip if it encounters a closing tag artifact
            content = content.split('</formula>')[0]
            return self._clean_string(content)

        # 3. Fallback: If code block exists
        code_matches = re.findall(r'```(?:python)?(.*?)```', response, re.DOTALL)
        if code_matches:
             # Take the last code block
             return self._clean_string(code_matches[-1])
             
        # 4. Last resort: Pattern matching for formula functions
        # Look for a line starting with typical functions
        lines = response.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^(ts_|rank\(|log\(|-1|\()', line):
                return self._clean_string(line)

        # If we failed to find a formula, return the current one rather than garbage text
        print("DEBUG: Failed to extract formula, returning current.")
        return current_formula

    def _extract_formulas(self, text):
        """Parses JSON list from LLM output."""
        if not text: return []
        
        # Remove reasoning traces if present (DeepSeek style)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        try:
            # Try to find JSON array block
            # Look for the last occurrence of brackets if there are multiple/nested
            matches = re.findall(r'\[.*?\]', text, re.DOTALL)
            if matches:
                 # Iterate backwards to find the first valid JSON list
                for match in reversed(matches):
                    try:
                        # cleanup markdown inside if any
                        clean_match = match.replace('\n', ' ').strip()
                        parsed = json.loads(clean_match)
                        if isinstance(parsed, list):
                            return parsed
                    except:
                        continue
            
            # Fallback: try to clean markdown code blocks
            code_match = re.search(r'```json(.*?)```', text, re.DOTALL)
            if code_match:
                 return json.loads(code_match.group(1))

            return []
        except Exception as e:
            print(f"Parsing Error: {e}")
            return []

    def _clean_string(self, text):
        if not text: return ""
        text = text.strip()
        text = text.replace("```python", "").replace("```", "").strip()
        # Aggressively strip outer quotes
        text = text.strip('"').strip("'")
        return text.strip()
