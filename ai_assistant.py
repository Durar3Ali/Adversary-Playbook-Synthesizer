"""
AI Assistant Module - Smart Alert Analyst
Acts as an intelligent security analyst using LLM capabilities
"""

import os
import json
from typing import Dict, Optional, Any
from datetime import datetime

# Try to import Google Generative AI (Gemini), but make it optional
try:
    import google.generativeai as genai  # type: ignore[import]
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # type: ignore[assignment]

# Shared system prompt for all LLM assistants
SYSTEM_PROMPT = """You are an expert cybersecurity analyst assistant specializing in alert analysis and incident response. 
Your role is to:
1. Analyze security alerts with deep understanding of threat patterns
2. Provide clear, actionable insights about alerts
3. Explain technical findings in accessible language
4. Recommend appropriate response actions
5. Answer questions about alerts, classifications, and recommended responses

You have access to:
- Alert classification results (Benign/Malignant)
- XAI explanations showing top contributing features
- Generated response playbooks for malicious alerts
- Complete alert metadata

Always be:
- Precise and technical when needed
- Clear and accessible for non-technical users
- Proactive in identifying risks
- Practical in recommendations"""

# Fallback: Use a simple rule-based assistant if LLM is not available
class SimpleAIAssistant:
    """Simple rule-based AI assistant as fallback"""
    
    def __init__(self):
        self.name = "Security Analyst Assistant"
        self.version = "1.0"
    
    def analyze_alert(self, alert_data: Dict, prediction: Dict, explanation: Dict, playbook: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze alert using rule-based logic"""
        
        label = prediction.get('label', 'Unknown')
        probability = prediction.get('probability', 0.0)
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'assistant_name': self.name,
            'alert_assessment': label,
            'confidence_level': 'HIGH' if abs(probability - 0.5) > 0.3 else 'MEDIUM',
            'key_findings': [],
            'recommendations': [],
            'risk_score': probability
        }
        
        # Analyze based on features
        top_features = explanation.get('top_features', [])
        
        if label == 'Malicious':
            analysis['key_findings'].append(f"Alert classified as malicious with {probability:.1%} confidence")
            
            # Check for specific indicators
            for feat in top_features[:3]:
                feat_name = feat.get('feature', '').lower()
                contribution = feat.get('contribution', 0)
                
                if abs(contribution) > 0.1:
                    if 'packet' in feat_name:
                        analysis['key_findings'].append("Unusual packet patterns detected")
                        analysis['recommendations'].append("Review network traffic for similar patterns")
                    elif 'port' in feat_name:
                        analysis['key_findings'].append("Suspicious port activity identified")
                        analysis['recommendations'].append("Verify if port usage is authorized")
                    elif 'bytes' in feat_name:
                        analysis['key_findings'].append("Anomalous data transfer detected")
                        analysis['recommendations'].append("Investigate data exfiltration possibilities")
            
            if probability > 0.8:
                analysis['recommendations'].append("Immediate containment recommended")
                analysis['recommendations'].append("Notify incident response team immediately")
        else:
            analysis['key_findings'].append(f"Alert appears benign ({probability:.1%} malicious probability)")
            analysis['recommendations'].append("Continue monitoring for similar patterns")
            analysis['recommendations'].append("No immediate action required")
        
        return analysis
    
    def answer_question(self, question: str, context: Dict) -> str:
        """Answer questions based on context"""
        question_lower = question.lower()
        
        if 'malicious' in question_lower or 'threat' in question_lower:
            label = context.get('prediction', {}).get('label', 'Unknown')
            if label == 'Malicious':
                return f"Yes, this alert has been classified as malicious. The system detected suspicious patterns that indicate a potential security threat. Immediate action is recommended."
            else:
                return "No, this alert appears to be benign. However, continue monitoring for similar patterns."
        
        elif 'probability' in question_lower or 'confidence' in question_lower:
            prob = context.get('prediction', {}).get('probability', 0.0)
            return f"The malicious probability is {prob:.1%}. This indicates a {'high' if prob > 0.7 else 'medium' if prob > 0.4 else 'low'} level of concern."
        
        elif 'playbook' in question_lower or 'response' in question_lower:
            if context.get('playbook'):
                return "A response playbook has been generated for this alert. It includes step-by-step instructions for containment, investigation, and remediation."
            else:
                return "No playbook is available for this alert. Playbooks are only generated for malicious alerts."
        
        elif 'explain' in question_lower or 'why' in question_lower:
            explanation = context.get('explanation', {}).get('explanation_text', 'No explanation available')
            return f"Based on the analysis: {explanation}"
        
        else:
            return "I can help you understand the alert classification, probability, explanations, and response recommendations. Please ask a specific question about the alert."


class GeminiAssistant:
    """LLM-powered AI assistant using Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini assistant.
        
        Parameters:
        -----------
        api_key : str, optional
            Gemini API key. If None, will try to get from environment variable GEMINI_API_KEY
        model : str
            Model to use (default: gemini-2.5-flash)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini API: {str(e)}. Please check your API key.")
        
        # Try to find an available model
        try:
            self.model_name = self._find_available_model(model)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✓ Successfully initialized Gemini with model: {self.model_name}")
        except Exception as e:
            # Provide helpful error message
            error_msg = f"Failed to initialize Gemini model '{model}': {str(e)}\n"
            error_msg += "This usually means:\n"
            error_msg += "1. The model name doesn't match what your API key supports\n"
            error_msg += "2. Your API key doesn't have access to Gemini models\n"
            error_msg += "3. The API key is invalid or expired\n"
            error_msg += "Please check your API key in Google Cloud Console."
            raise ValueError(error_msg)
        
        self.name = "Gemini Security Analyst Assistant"
        self.version = "2.0"
    
    def _find_available_model(self, preferred_model: str) -> str:
        """
        Find an available model by trying the preferred model and fallbacks.
        
        Parameters:
        -----------
        preferred_model : str
            Preferred model name to try first
            
        Returns:
        --------
        str : Available model name
        """
        # List of models to try in order of preference
        # Based on diagnostic: available models are gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, etc.
        models_to_try = [
            preferred_model,
            "gemini-2.5-flash",  # Latest stable flash model
            "gemini-2.0-flash",   # Stable 2.0 flash model
            "gemini-2.5-pro",     # Latest pro model
            "gemini-flash-latest", # Latest flash (auto-updates)
            "gemini-pro-latest",   # Latest pro (auto-updates)
            "gemini-1.5-flash",    # Fallback (may not work)
            "gemini-1.5-pro",      # Fallback (may not work)
            "gemini-pro",          # Legacy fallback
        ]
        
        # Try to list available models first (if API supports it)
        try:
            available_models_list = list(genai.list_models())
            if available_models_list:
                # Extract model names
                available_names = []
                for m in available_models_list:
                    model_name = m.name if hasattr(m, 'name') else str(m)
                    # Check if it supports generateContent
                    if hasattr(m, 'supported_generation_methods'):
                        if 'generateContent' in m.supported_generation_methods:
                            available_names.append(model_name)
                    else:
                        available_names.append(model_name)
                
                # Try preferred model first if it's in the list
                for model in models_to_try:
                    for avail_name in available_names:
                        # Match with or without 'models/' prefix
                        clean_avail = avail_name.replace('models/', '')
                        clean_model = model.replace('models/', '')
                        if clean_model == clean_avail or model == avail_name:
                            try:
                                # Test if we can create the model
                                test_model = genai.GenerativeModel(clean_model)
                                return clean_model
                            except Exception as e:
                                continue
                
                # If preferred not found, try first available flash model
                for avail_name in available_names:
                    if 'flash' in avail_name.lower():
                        clean_name = avail_name.replace('models/', '')
                        try:
                            genai.GenerativeModel(clean_name)
                            return clean_name
                        except:
                            continue
        except Exception as e:
            # If list_models fails, continue with direct testing
            pass
        
        # Fallback: try models one by one directly
        last_error = None
        for model in models_to_try:
            try:
                # Remove models/ prefix for testing
                clean_model = model.replace('models/', '')
                test_model = genai.GenerativeModel(clean_model)
                # If we get here, the model works
                return clean_model
            except Exception as e:
                last_error = e
                continue
        
        # If nothing works, raise error with details
        error_msg = f"Could not find an available Gemini model. Tried: {', '.join(models_to_try)}."
        if last_error:
            error_msg += f" Last error: {str(last_error)}"
        error_msg += " Please check your API key permissions and ensure it has access to Gemini models."
        raise ValueError(error_msg)
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the AI assistant"""
        return SYSTEM_PROMPT
    
    def analyze_alert(self, alert_data: Dict, prediction: Dict, explanation: Dict, playbook: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze alert using Gemini"""
        
        # Create prompt
        top_features = explanation.get('top_features', [])[:5]
        features_text = "\n".join([
            f"- {f.get('feature', 'Unknown')}: {f.get('value', 'N/A')} (contribution: {f.get('contribution', 0):+.4f})"
            for f in top_features
        ])
        
        playbook_info = (
            f"PLAYBOOK AVAILABLE: Yes - Threat Level: {playbook.get('threat_level', 'Unknown')}"
            if playbook and playbook.get('playbook_required')
            else "PLAYBOOK: Not required (benign alert)"
        )
        
        prompt = f"""{self._create_system_prompt()}

Analyze this security alert and provide your expert assessment:

ALERT DATA:
{json.dumps(alert_data, indent=2)}

CLASSIFICATION:
- Label: {prediction.get('label', 'Unknown')}
- Malicious Probability: {prediction.get('probability', 0.0):.2%}

EXPLANATION:
{explanation.get('explanation_text', 'No explanation available')}

Top Contributing Features:
{features_text}

{playbook_info}

Please provide:
1. Your assessment of the alert
2. Key findings and concerns
3. Immediate recommendations
4. Risk level assessment
5. Any additional insights

Format your response as a structured analysis."""
        
        try:
            generation_config = {
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            analysis_text = response.text
            
            return {
                'timestamp': datetime.now().isoformat(),
                'assistant_name': self.name,
                'assistant_version': self.version,
                'analysis': analysis_text,
                'alert_assessment': prediction.get('label', 'Unknown'),
                'risk_score': prediction.get('probability', 0.0),
                'model_used': self.model_name
            }
        
        except Exception as e:
            # Fallback to simple assistant on error
            simple_assistant = SimpleAIAssistant()
            result = simple_assistant.analyze_alert(alert_data, prediction, explanation)
            result['error'] = f"Gemini analysis failed: {str(e)}. Using fallback analysis."
            return result
    
    def answer_question(self, question: str, context: Dict) -> str:
        """Answer user questions using Gemini"""
        
        # Prepare context summary
        context_summary = f"""
ALERT CONTEXT:
- Classification: {context.get('prediction', {}).get('label', 'Unknown')}
- Probability: {context.get('prediction', {}).get('probability', 0.0):.2%}
- Explanation: {context.get('explanation', {}).get('explanation_text', 'N/A')}
- Has Playbook: {'Yes' if context.get('playbook') else 'No'}
"""
        
        prompt = f"""{self._create_system_prompt()}

User Question: {question}

{context_summary}

Please provide a clear, helpful answer to the user's question based on the alert context."""
        
        try:
            generation_config = {
                "temperature": 0.5,
                "max_output_tokens": 500,
            }
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        
        except Exception as e:
            # Fallback
            simple_assistant = SimpleAIAssistant()
            return simple_assistant.answer_question(question, context) + f"\n\n(Note: Gemini unavailable, using fallback. Error: {str(e)})"


# Factory function to create appropriate assistant
def create_assistant(use_llm: bool = False, llm_provider: str = "gemini", api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
    """
    Create an AI assistant instance.
    
    Parameters:
    -----------
    use_llm : bool
        Whether to use LLM (requires API key)
    llm_provider : str
        LLM provider: "gemini" (default: "gemini")
    api_key : str, optional
        API key (required if use_llm=True)
    model : str
        Model to use if LLM (default: gemini-2.5-flash)
    
    Returns:
    --------
    Assistant instance (GeminiAssistant or SimpleAIAssistant)
    """
    if use_llm:
        if llm_provider.lower() == "gemini":
            try:
                return GeminiAssistant(api_key=api_key, model=model)
            except (ImportError, ValueError) as e:
                print(f"Warning: Could not initialize Gemini assistant: {e}")
                print("Falling back to simple rule-based assistant.")
                return SimpleAIAssistant()
        else:
            print(f"Warning: Unknown LLM provider '{llm_provider}'. Only 'gemini' is supported.")
            print("Falling back to simple rule-based assistant.")
            return SimpleAIAssistant()
    else:
        return SimpleAIAssistant()