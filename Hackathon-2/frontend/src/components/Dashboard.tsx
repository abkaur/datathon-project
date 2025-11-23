/**
 * React Frontend - Crop Yield Prediction Dashboard
 * UN SDG 2: Zero Hunger
 */

import React, { useState, useCallback } from 'react';
import axios from 'axios';

// Types
interface PredictionRequest {
  latitude: number;
  longitude: number;
  date: string;
  crop_type: string;
  temperature: number;
  precipitation: number;
  humidity: number;
  soil_moisture: number;
}

interface PredictionResponse {
  prediction_id: string;
  yield_estimate: number;
  confidence_score: number;
  rmse: number;
  model_version: string;
  timestamp: string;
  crop_type: string;
  input_features: Record<string, number>;
}

interface SummaryResponse {
  summary: string;
  recommendations: string[];
  confidence_explanation: string;
  risk_factors: string[];
}

// API Client
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
});

// Components
export const PredictionDashboard: React.FC = () => {
  const [formData, setFormData] = useState<PredictionRequest>({
    latitude: 40.7128,
    longitude: -74.0060,
    date: new Date().toISOString().split('T')[0],
    crop_type: 'maize',
    temperature: 25,
    precipitation: 100,
    humidity: 65,
    soil_moisture: 0.4,
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle form input changes
  const handleInputChange = useCallback((
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: isNaN(Number(value)) ? value : Number(value),
    }));
  }, []);

  // Submit prediction request
  const handlePredictClick = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.post<PredictionResponse>(
        '/predict',
        formData
      );
      
      setPrediction(response.data);
      
      // Auto-generate summary after prediction
      await generateSummary(response.data.prediction_id);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Prediction failed';
      setError(errorMsg);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  }, [formData]);

  // Generate AI summary
  const generateSummary = useCallback(async (predictionId: string) => {
    try {
      const response = await apiClient.post<SummaryResponse>(
        '/summary',
        { prediction_id: predictionId, include_recommendations: true }
      );
      setSummary(response.data);
    } catch (err) {
      console.error('Summary generation error:', err);
    }
  }, []);

  // Crop type options
  const cropTypes = ['maize', 'wheat', 'rice', 'soybean', 'potato', 'sugarcane'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-4xl">🌾</span>
            <h1 className="text-4xl font-bold text-gray-800">
              Crop Yield Prediction System
            </h1>
          </div>
          <p className="text-gray-600">
            AI-powered yield prediction using satellite imagery and weather data
          </p>
          <div className="mt-2 inline-block px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
            UN SDG 2: Zero Hunger
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Form - Left Column */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6 h-full">
              <h2 className="text-xl font-bold mb-4 text-gray-800">
                📋 Prediction Parameters
              </h2>

              <div className="space-y-4">
                {/* Location Inputs */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Latitude
                  </label>
                  <input
                    type="number"
                    name="latitude"
                    value={formData.latitude}
                    onChange={handleInputChange}
                    step="0.01"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={loading}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Longitude
                  </label>
                  <input
                    type="number"
                    name="longitude"
                    value={formData.longitude}
                    onChange={handleInputChange}
                    step="0.01"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={loading}
                  />
                </div>

                {/* Date Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Prediction Date
                  </label>
                  <input
                    type="date"
                    name="date"
                    value={formData.date}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={loading}
                  />
                </div>

                {/* Crop Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Crop Type
                  </label>
                  <select
                    name="crop_type"
                    value={formData.crop_type}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={loading}
                  >
                    {cropTypes.map(crop => (
                      <option key={crop} value={crop}>
                        {crop.charAt(0).toUpperCase() + crop.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Weather Parameters */}
                <div className="pt-2 border-t">
                  <h3 className="font-medium text-gray-700 mb-3">Weather Data</h3>

                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">
                        Temperature (°C)
                      </label>
                      <input
                        type="number"
                        name="temperature"
                        value={formData.temperature}
                        onChange={handleInputChange}
                        step="0.1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                        disabled={loading}
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-700 mb-1">
                        Precipitation (mm)
                      </label>
                      <input
                        type="number"
                        name="precipitation"
                        value={formData.precipitation}
                        onChange={handleInputChange}
                        min="0"
                        step="1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                        disabled={loading}
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-700 mb-1">
                        Humidity (%)
                      </label>
                      <input
                        type="number"
                        name="humidity"
                        value={formData.humidity}
                        onChange={handleInputChange}
                        min="0"
                        max="100"
                        step="0.1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                        disabled={loading}
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-700 mb-1">
                        Soil Moisture (0-1)
                      </label>
                      <input
                        type="number"
                        name="soil_moisture"
                        value={formData.soil_moisture}
                        onChange={handleInputChange}
                        min="0"
                        max="1"
                        step="0.01"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                        disabled={loading}
                      />
                    </div>
                  </div>
                </div>

                {/* Error Message */}
                {error && (
                  <div className="p-3 bg-red-100 text-red-700 rounded-md text-sm">
                    ⚠️ {error}
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handlePredictClick}
                  disabled={loading}
                  className="w-full mt-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded-md transition-colors flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <span className="animate-spin">⚙️</span>
                      Predicting...
                    </>
                  ) : (
                    <>
                      🚀 Generate Prediction
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Results - Right Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Prediction Results */}
            {prediction && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-800">
                  📊 Prediction Results
                </h2>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {/* Yield Estimate */}
                  <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Yield Estimate</p>
                    <p className="text-2xl font-bold text-green-700">
                      {prediction.yield_estimate.toFixed(0)}
                    </p>
                    <p className="text-xs text-gray-500">kg/ha</p>
                  </div>

                  {/* Confidence Score */}
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Confidence</p>
                    <p className="text-2xl font-bold text-blue-700">
                      {(prediction.confidence_score * 100).toFixed(0)}%
                    </p>
                    <div className="mt-2 w-full bg-blue-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${prediction.confidence_score * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Model Info */}
                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Model Version</p>
                    <p className="text-lg font-bold text-purple-700">
                      {prediction.model_version}
                    </p>
                    <p className="text-xs text-gray-500 mt-2">RMSE: {prediction.rmse.toFixed(0)}</p>
                  </div>

                  {/* Crop Type */}
                  <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Crop Type</p>
                    <p className="text-lg font-bold text-yellow-700 capitalize">
                      {prediction.crop_type}
                    </p>
                  </div>

                  {/* NDVI */}
                  <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">NDVI</p>
                    <p className="text-2xl font-bold text-orange-700">
                      {(prediction.input_features.ndvi || 0.65).toFixed(2)}
                    </p>
                  </div>

                  {/* Timestamp */}
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Predicted At</p>
                    <p className="text-xs font-mono text-gray-700">
                      {new Date(prediction.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* AI Summary */}
            {summary && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-800">
                  🤖 AI-Generated Summary
                </h2>

                <div className="space-y-4">
                  {/* Summary Text */}
                  <div className="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                    <p className="text-gray-700 leading-relaxed">
                      {summary.summary}
                    </p>
                  </div>

                  {/* Confidence Explanation */}
                  <div>
                    <h3 className="font-medium text-gray-700 mb-2">💡 Model Confidence</h3>
                    <p className="text-gray-600 text-sm">
                      {summary.confidence_explanation}
                    </p>
                  </div>

                  {/* Recommendations */}
                  {summary.recommendations.length > 0 && (
                    <div>
                      <h3 className="font-medium text-gray-700 mb-2">✅ Recommendations</h3>
                      <ul className="space-y-2">
                        {summary.recommendations.map((rec, idx) => (
                          <li key={idx} className="flex gap-2 text-sm text-gray-700">
                            <span className="text-green-600">→</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Risk Factors */}
                  {summary.risk_factors.length > 0 && (
                    <div>
                      <h3 className="font-medium text-gray-700 mb-2">⚠️ Risk Factors</h3>
                      <ul className="space-y-2">
                        {summary.risk_factors.map((risk, idx) => (
                          <li key={idx} className="flex gap-2 text-sm text-gray-700">
                            <span className="text-red-600">⚠</span>
                            <span>{risk}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Empty State */}
            {!prediction && !loading && (
              <div className="bg-white rounded-lg shadow-md p-12 text-center">
                <span className="text-6xl block mb-4">📍</span>
                <p className="text-gray-600">
                  Fill in the parameters and click "Generate Prediction" to see results
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p>
            🌱 Supporting UN SDG 2: Zero Hunger through data-driven agricultural decisions
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionDashboard;
