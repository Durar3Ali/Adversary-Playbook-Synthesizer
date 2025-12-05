"""
FastAPI backend for Cyber Security Alert Classification
Provides prediction and explanation endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import sys
import os

# Import prediction functions
from ml_model import predict_alert, explain_alert

app = FastAPI(
    title="Cyber Security Alert Classifier API",
    description="API for classifying cyber security alerts as Normal or Malicious with explainability",
    version="1.0.0"
)


class AlertRequest(BaseModel):
    """Request model for alert prediction"""
    Source_IP: Optional[str] = None
    Destination_IP: Optional[str] = None
    Protocol: Optional[str] = None
    Packet_Length: Optional[float] = None
    Duration: Optional[float] = None
    Source_Port: Optional[int] = None
    Destination_Port: Optional[int] = None
    Bytes_Sent: Optional[float] = None
    Bytes_Received: Optional[float] = None
    Flags: Optional[str] = None
    Flow_Packets_s: Optional[float] = None
    Flow_Bytes_s: Optional[float] = None
    Avg_Packet_Size: Optional[float] = None
    Total_Fwd_Packets: Optional[int] = None
    Total_Bwd_Packets: Optional[int] = None
    Fwd_Header_Length: Optional[int] = None
    Bwd_Header_Length: Optional[int] = None
    Sub_Flow_Fwd_Bytes: Optional[float] = None
    Sub_Flow_Bwd_Bytes: Optional[float] = None
    Inbound: Optional[int] = None
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "Source_IP": "192.168.0.1",
                "Destination_IP": "10.0.0.3",
                "Protocol": "TCP",
                "Packet_Length": 1500,
                "Duration": 2.5,
                "Source_Port": 80,
                "Destination_Port": 443,
                "Bytes_Sent": 1000,
                "Bytes_Received": 2000,
                "Flags": "SYN",
                "Flow_Packets_s": 30.5,
                "Flow_Bytes_s": 1500.0,
                "Avg_Packet_Size": 512,
                "Total_Fwd_Packets": 25,
                "Total_Bwd_Packets": 30,
                "Fwd_Header_Length": 256,
                "Bwd_Header_Length": 256,
                "Sub_Flow_Fwd_Bytes": 800,
                "Sub_Flow_Bwd_Bytes": 1200,
                "Inbound": 1
            }
        }
    
    def to_dict(self):
        """Convert to dict with proper field names matching CSV columns"""
        d = self.dict(exclude_none=True)
        # Map Python field names to CSV column names (with slashes)
        if 'Flow_Packets_s' in d:
            d['Flow_Packets/s'] = d.pop('Flow_Packets_s')
        if 'Flow_Bytes_s' in d:
            d['Flow_Bytes/s'] = d.pop('Flow_Bytes_s')
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Create from dict, handling field name mapping"""
        # Map CSV column names (with slashes) to Python field names
        mapped_dict = {}
        for key, value in d.items():
            if key == 'Flow_Packets/s':
                mapped_dict['Flow_Packets_s'] = value
            elif key == 'Flow_Bytes/s':
                mapped_dict['Flow_Bytes_s'] = value
            else:
                mapped_dict[key] = value
        return cls(**mapped_dict)


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int
    probability: float
    label: str


class FeatureContribution(BaseModel):
    """Model for feature contribution in explanation"""
    feature: str
    value: Any
    contribution: float


class ExplanationResponse(BaseModel):
    """Response model for prediction with explanation"""
    prediction: int
    probability: float
    label: str
    top_features: List[FeatureContribution]
    explanation_text: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(alert: AlertRequest):
    """
    Predict if an alert is Normal (0) or Malicious (1)
    
    Returns prediction, probability, and label without explanation
    """
    try:
        # Convert Pydantic model to dict with proper field names
        alert_dict = alert.to_dict()
        
        # Make prediction
        result = predict_alert(alert_dict)
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            label=result['label']
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {str(e)}. Please train the model first by running 'python ml_model.py'"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing alert: {str(e)}"
        )


@app.post("/predict_with_explanation", response_model=ExplanationResponse)
async def predict_with_explanation(alert: AlertRequest, top_k: int = 5):
    """
    Predict if an alert is Normal (0) or Malicious (1) with explanation
    
    Returns prediction, probability, label, top contributing features, and explanation text
    """
    try:
        # Convert Pydantic model to dict with proper field names
        alert_dict = alert.to_dict()
        
        # Make prediction with explanation
        result = explain_alert(alert_dict, top_k=top_k)
        
        # Convert top_features to FeatureContribution models
        top_features = [
            FeatureContribution(
                feature=feat['feature'],
                value=feat['value'],
                contribution=feat['contribution']
            )
            for feat in result['top_features']
        ]
        
        return ExplanationResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            label=result['label'],
            top_features=top_features,
            explanation_text=result['explanation_text']
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {str(e)}. Please train the model first by running 'python ml_model.py'"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing alert: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

